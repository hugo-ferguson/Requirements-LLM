"""Acceptance criteria generation: retrieve context, generate, score and rank."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

from app.config import Settings
from app.generation.models import CandidateGroup, EnsembleResult, GeneratedCriterion
from app.generation.orchestrator import (
    pool_criteria,
    run_ensemble,
    run_title_anchored_ensemble,
)
from app.generation.prompts import GenerationDeps
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionAlternative,
    AcceptanceCriterionScores,
    ConversationMessage,
)
from app.services.agents import GenerationError
from app.services.scoring import CandidateScore, Scorer, render_ac_candidate
from app.services.scoring import score_candidates as default_score_candidates

logger = logging.getLogger(__name__)


class RetrievedChunk(Protocol):
    """A retrieved context chunk; only its text is read."""

    content: str


class ContextRetriever(Protocol):
    """The part of `IngestService` that generation uses, so tests can fake it."""

    def search(self, query: str, top_k: int = 5) -> Sequence[RetrievedChunk]:
        """Return the `top_k` stored chunks most similar to `query`."""
        ...


class NoUserStoryError(ValueError):
    """Raised when a request has no user story to generate from."""


def compose_user_story(messages: list[ConversationMessage]) -> str:
    """Join the user messages and their attachments into the story text.

    Assistant replies are left out, as they would bias generation.

    Raises:
        NoUserStoryError: No user turn has any text or attachment content.
    """
    parts: list[str] = []

    for message in messages:
        if message.role != "user":
            continue
        if message.text.strip():
            parts.append(message.text.strip())
        for attachment in message.attachments:
            if attachment.content.strip():
                parts.append(
                    f"--- Attached file: {attachment.filename} ---\n"
                    f"{attachment.content.strip()}"
                )

    story = "\n\n".join(parts).strip()
    if not story:
        raise NoUserStoryError(
            "No user story provided. Add a message describing the requirement "
            "before generating acceptance criteria."
        )
    return story


class GenerationService:
    """The ensemble generation pipeline for one request."""

    def __init__(
        self,
        settings: Settings,
        context_retriever: ContextRetriever | None = None,
        scorer: Scorer | None = None,
    ):
        """Store the settings, the RAG retriever and the scorer.

        Args:
            settings: App settings.
            context_retriever: Source of project context. Optional.
            scorer: Replaces the voting layer, for tests.
        """
        self.settings = settings
        self.context_retriever = context_retriever
        self.scorer: Scorer = scorer or default_score_candidates

    async def retrieve_context(self, query: str) -> list[str]:
        """Fetch the top-K chunks relevant to the story (R11), or none on failure."""
        if self.context_retriever is None or self.settings.rag_top_k <= 0:
            return []

        try:
            chunks = await asyncio.to_thread(
                self.context_retriever.search, query, self.settings.rag_top_k
            )
        except Exception:
            logger.exception("RAG retrieval failed; generating without project context")
            return []

        return [chunk.content for chunk in chunks]

    async def score(self, prompt: str, candidates: list[str]) -> list[CandidateScore]:
        """Return one score per candidate, in the order given."""
        return await asyncio.to_thread(
            self.scorer,
            prompt=prompt,
            candidates=candidates,
            settings=self.settings,
        )

    async def generate(
        self,
        messages: list[ConversationMessage],
        *,
        feedback: str | None = None,
    ) -> list[AcceptanceCriterion]:
        """Run the full pipeline: retrieve, generate, score, rank.

        Args:
            messages: The session's conversation, oldest first.
            feedback: Reviewer feedback when regenerating (R20).

        Returns:
            The criteria, with `id` set to list position.

        Raises:
            NoUserStoryError: The conversation has no user story in it.
            GenerationError: Every generation agent failed.
        """
        story = compose_user_story(messages)
        context_chunks = await self.retrieve_context(story)

        deps = GenerationDeps(
            user_story=story,
            context_chunks=context_chunks,
            feedback=feedback,
        )

        groups: list[CandidateGroup] = []
        ensemble: EnsembleResult | None = None
        if self.settings.generation_title_anchored:
            ensemble, groups = await run_title_anchored_ensemble(
                deps,
                max_criteria=self.settings.generation_max_criteria,
                timeout_seconds=self.settings.generation_timeout_seconds,
            )
            if not groups and ensemble.successful:
                logger.warning(
                    "title-anchored generation produced no groups; "
                    "falling back to the flat ensemble"
                )
        if ensemble is None or not groups:
            ensemble = await run_ensemble(
                deps,
                max_criteria=self.settings.generation_max_criteria,
                timeout_seconds=self.settings.generation_timeout_seconds,
            )

        self._log_ensemble(ensemble)

        if not ensemble.successful:
            reasons = "; ".join(
                f"{result.agent_id}: {result.error}" for result in ensemble.failed
            )
            raise GenerationError(f"Every generation agent failed. {reasons}")

        if groups:
            return await self._generate_grouped(ensemble, groups)

        pooled = pool_criteria(ensemble)
        criteria = [criterion for _, criterion in pooled]

        if not self.settings.generation_enable_voting:
            return _to_wire_models(criteria, scores=None)

        candidate_texts = [
            render_ac_candidate(
                criterion.title, criterion.given, criterion.when, criterion.then
            )
            for criterion in criteria
        ]
        scores = await self.score(ensemble.prompt, candidate_texts)
        return _to_wire_models(criteria, scores=scores)

    async def _generate_grouped(
        self, ensemble: EnsembleResult, groups: list[CandidateGroup]
    ) -> list[AcceptanceCriterion]:
        """Score every candidate in one call, then rank within each group."""
        flat = [
            (group, candidate) for group in groups for candidate in group.candidates
        ]

        if self.settings.generation_enable_voting:
            texts = [
                render_ac_candidate(
                    candidate.criterion.title,
                    candidate.criterion.given,
                    candidate.criterion.when,
                    candidate.criterion.then,
                )
                for _, candidate in flat
            ]
            scores = await self.score(ensemble.prompt, texts)
            for (_, candidate), score in zip(flat, scores, strict=True):
                candidate.relevance = score.relevance
                candidate.correctness = score.correctness
                candidate.understandability = score.understandability
                candidate.coverage = score.coverage
                candidate.overall_score = score.overall
                candidate.scoring_failed = score.failed

        for group in groups:
            group.rank()

        contested = [group for group in groups if len(group.candidates) > 1]
        logger.info(
            "grouped %d criteria from %d candidates; %d contested, %d unrated",
            len(groups),
            len(flat),
            len(contested),
            sum(len(group.unrated) for group in groups),
        )
        for group in contested:
            if group.winner.scoring_failed:
                logger.warning(
                    "no candidate for %r was rated; the winner was chosen by "
                    "tie-break, not by score",
                    group.title,
                )

        # Keep the anchor's order, not score order.
        return _groups_to_wire_models(groups)

    @staticmethod
    def _log_ensemble(ensemble: EnsembleResult) -> None:
        """Log one line per agent: how many criteria, or why it failed."""
        for result in ensemble.results:
            if result.ok:
                logger.info(
                    "agent=%s model=%s criteria=%d duration_ms=%d",
                    result.agent_id,
                    result.model,
                    len(result.criteria),
                    result.duration_ms,
                )
            else:
                logger.warning(
                    "agent=%s model=%s FAILED after %dms: %s",
                    result.agent_id,
                    result.model,
                    result.duration_ms,
                    result.error,
                )


def _unscored() -> AcceptanceCriterionScores:
    """Return placeholder scores for when voting is switched off."""
    return AcceptanceCriterionScores(
        relevance=0.0, correctness=0.0, understandability=0.0, coverage=0.0
    )


def _groups_to_wire_models(groups: list[CandidateGroup]) -> list[AcceptanceCriterion]:
    """Convert each ranked group to its winner, with the rest as `alternatives`."""
    wire: list[AcceptanceCriterion] = []

    for index, group in enumerate(groups, start=1):
        if not group.candidates:
            continue
        winner = group.winner
        wire.append(
            AcceptanceCriterion(
                id=index,
                # The anchor's title, whichever agent won.
                title=group.title,
                given=winner.criterion.given,
                when=winner.criterion.when,
                then=winner.criterion.then,
                scores=AcceptanceCriterionScores(
                    relevance=winner.relevance,
                    correctness=winner.correctness,
                    understandability=winner.understandability,
                    coverage=winner.coverage,
                ),
                overall_score=winner.overall_score,
                status="pending",
                source_agent=winner.agent_id,
                alternatives=[
                    AcceptanceCriterionAlternative(
                        given=alternative.criterion.given,
                        when=alternative.criterion.when,
                        then=alternative.criterion.then,
                        scores=AcceptanceCriterionScores(
                            relevance=alternative.relevance,
                            correctness=alternative.correctness,
                            understandability=alternative.understandability,
                            coverage=alternative.coverage,
                        ),
                        overall_score=alternative.overall_score,
                        source_agent=alternative.agent_id,
                    )
                    for alternative in group.alternatives
                ],
            )
        )

    return wire


def _to_wire_models(
    criteria: list[GeneratedCriterion], scores: list[CandidateScore] | None
) -> list[AcceptanceCriterion]:
    """Rank best-first and convert to the schema the frontend reads.

    Rank before saving, as list order is stored as `position` (R14, R17).
    """
    if scores is None:
        paired = [(criterion, _unscored(), 0.0) for criterion in criteria]
    else:
        paired = [
            (
                criterion,
                AcceptanceCriterionScores(
                    relevance=score.relevance,
                    correctness=score.correctness,
                    understandability=score.understandability,
                    coverage=score.coverage,
                ),
                score.overall,
            )
            for criterion, score in zip(criteria, scores, strict=True)
        ]

    ranked = sorted(paired, key=lambda item: item[2], reverse=True)

    return [
        AcceptanceCriterion(
            id=index,
            title=criterion.title,
            given=criterion.given,
            when=criterion.when,
            then=criterion.then,
            scores=criterion_scores,
            overall_score=overall,
            status="pending",
        )
        for index, (criterion, criterion_scores, overall) in enumerate(ranked, start=1)
    ]
