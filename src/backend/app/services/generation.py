"""Business logic for ensemble acceptance criteria generation.

Sits at the same level as the other services: it knows nothing about HTTP or
about SQLModel, and it is the only place the stages of the pipeline are
sequenced together.

    conversation -> RAG retrieval -> ensemble generation -> scoring -> ranked ACs

Scoring is delegated to `app.services.scoring`, which is shared with AC
regeneration and UAT generation. That module owns the voting-layer contract:
it matches results back to candidates by exact text (the voting layer sorts its
output by score, so position is not stable) and clamps rubric values onto the
app's 0-5 scale. This layer's job is the ensemble, not the scoring.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from app.config import Settings
from app.generation.models import EnsembleResult
from app.generation.orchestrator import pool_criteria, run_ensemble
from app.generation.prompts import GenerationDeps
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
)
from app.services.agents import GenerationError
from app.services.scoring import (
    Scorer,
    render_ac_candidate,
    score_candidates as default_score_candidates,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ContextRetriever(Protocol):
    """The slice of `IngestService` this layer actually needs.

    Depending on the shape rather than the concrete class keeps the generation
    layer testable without a database, and means the RAG implementation can
    change underneath without touching this file.
    """

    def search(self, query: str, k: int = 5): ...


class NoUserStoryError(ValueError):
    """Raised when a generation request contains nothing to generate from."""


def compose_user_story(messages: list[ConversationMessage]) -> str:
    """Fold the conversation into the single story text the agents receive.

    Only user turns are used — assistant replies are the system talking to
    itself and would bias generation. Attachment text is appended after the
    messages so uploaded requirements documents are visible to the agents even
    before retrieval runs.
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
    """Runs the ensemble generation pipeline for one request."""

    def __init__(
        self,
        settings: Settings,
        ingest_service: ContextRetriever | None = None,
        scorer: Scorer | None = None,
    ):
        self.settings = settings
        # Optional so the service can run without a vector store configured;
        # generation then proceeds on the user story alone.
        self.ingest = ingest_service
        # Injectable so tests can score without reaching the voting layer.
        self.scorer: Scorer = scorer or default_score_candidates

    async def retrieve_context(self, query: str) -> list[str]:
        """Pull the top-K most relevant chunks for this story (backlog R11).

        The embedding call and the pgvector query are both synchronous and
        network- or CPU-bound, so they run in a worker thread rather than
        blocking the event loop the agents are running on.
        """
        if self.ingest is None or self.settings.rag_top_k <= 0:
            return []

        try:
            chunks = await asyncio.to_thread(
                self.ingest.search, query, self.settings.rag_top_k
            )
        except Exception:  # noqa: BLE001 - retrieval is best-effort
            logger.exception(
                "RAG retrieval failed; generating without project context")
            return []

        return [chunk.content for chunk in chunks]

    async def score(
        self, prompt: str, candidates: list[str]
    ) -> list:
        """Score candidates through the shared voting seam.

        `score_candidates` is synchronous and calls `asyncio.run` internally,
        so it cannot be awaited directly from inside a running loop — it goes
        through a worker thread. It never raises: a scoring outage returns
        all-zero scores rather than losing the generated criteria.
        """
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
        """Full pipeline: retrieve, generate, score, rank."""
        story = compose_user_story(messages)
        context_chunks = await self.retrieve_context(story)

        deps = GenerationDeps(
            user_story=story,
            context_chunks=context_chunks,
            feedback=feedback,
        )

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

        pooled = pool_criteria(ensemble)
        criteria = [criterion for _, criterion in pooled]

        if not self.settings.generation_enable_voting:
            return _to_wire_models(criteria, scores=None)

        candidate_texts = [
            render_ac_candidate(c.title, c.given, c.when, c.then) for c in criteria
        ]
        scores = await self.score(ensemble.prompt, candidate_texts)
        return _to_wire_models(criteria, scores=scores)

    @staticmethod
    def _log_ensemble(ensemble: EnsembleResult) -> None:
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
    return AcceptanceCriterionScores(
        relevance=0.0, correctness=0.0, understandability=0.0, coverage=0.0
    )


def _to_wire_models(criteria: list, scores: list | None) -> list[AcceptanceCriterion]:
    """Rank highest-first and convert to the schema the frontend consumes.

    `id` is positional only — real ids are assigned by
    `AcceptanceCriteriaRepository.persist_batch`, which uses list order for
    `position`, so ranking has to happen before persistence (backlog R14, R17).
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
            id=index + 1,
            title=criterion.title,
            given=criterion.given,
            when=criterion.when,
            then=criterion.then,
            scores=scores_model,
            overall_score=overall,
            status="pending",
        )
        for index, (criterion, scores_model, overall) in enumerate(ranked, start=0)
    ]
