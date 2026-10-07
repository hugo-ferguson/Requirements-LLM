import asyncio
from collections.abc import Awaitable, Callable

from pydantic_ai import Agent

from app.config import Settings, settings as default_settings
from app.generation.models import GeneratedUatCase, UatAgentResult, UatCandidateGroup
from app.generation.orchestrator import run_uat_ensemble
from app.models_acceptance_criteria import AcceptanceCriterionRecord
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
    ConversationRequest,
)
from app.models_uat_cases import (
    UatApplyApprovedRequest,
    UatCase,
    UatCaseAlternative,
    UatCaseCandidateRecord,
    UatCaseGroup,
    UatCaseGroupsResult,
    UatCaseRecord,
    UatCaseScores,
    UatCaseStatusUpdate,
    UatCaseTextUpdate,
    UatRegenerateSelectedRequest,
    UatRegenerateSelectedResponse,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.agents import GenerationError, build_agent
from app.services.conversation import render_conversation
from app.services.scoring import Scorer, render_ac_candidate, render_uat_candidate, score_candidates


# `GeneratedUatCase` is re-exported from app.generation.models: the
# regeneration agent below still returns one, and tests import it from here.
UatEnsemble = Callable[..., Awaitable[tuple[list[UatAgentResult], list[UatCandidateGroup]]]]


UAT_REGENERATION_SYSTEM_PROMPT = """\
You are a QA engineer refining ONE existing UAT test case using a reviewer's
follow-up feedback.

You will be given the parent acceptance criterion, the test case as it
stands now, and a conversation of additional context or feedback from the
reviewer. Produce exactly one improved test case covering the same
underlying scenario — do not invent an unrelated new one. If the reviewer's
feedback conflicts with what's given, follow the feedback; it is the more
recent and more specific instruction.
"""


def _to_acceptance_criterion(record: AcceptanceCriterionRecord) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        id=record.id,
        title=record.title,
        given=record.given,
        when=record.when,
        then=record.then,
        scores=AcceptanceCriterionScores(
            relevance=record.relevance,
            correctness=record.correctness,
            understandability=record.understandability,
            coverage=record.coverage,
        ),
        overall_score=record.overall_score,
        status=record.status,
    )


def _to_uat_case(
    record: UatCaseRecord, candidates: list[UatCaseCandidateRecord] | None = None
) -> UatCase:
    """Wire model for one case, with the other models' versions attached."""
    candidates = candidates or []
    winner = next((c for c in candidates if c.is_winner), None)
    return UatCase(
        id=record.id,
        ac_id=record.ac_id,
        title=record.title,
        description=record.description,
        scores=UatCaseScores(
            relevance=record.relevance,
            correctness=record.correctness,
            understandability=record.understandability,
            coverage=record.coverage,
        ),
        overall_score=record.overall_score,
        status=record.status,
        source_agent=winner.source_agent if winner else None,
        alternatives=[
            UatCaseAlternative(
                candidate_id=c.id,
                description=c.description,
                scores=UatCaseScores(
                    relevance=c.relevance,
                    correctness=c.correctness,
                    understandability=c.understandability,
                    coverage=c.coverage,
                ),
                overall_score=c.overall_score,
                source_agent=c.source_agent,
            )
            for c in candidates
            if not c.is_winner
        ],
    )


def _build_regeneration_prompt(
    ac: AcceptanceCriterionRecord, target: UatCaseRecord, data: UatRegenerateSelectedRequest
) -> str:
    parts = [
        "Parent acceptance criterion:",
        render_ac_candidate(ac.title, ac.given, ac.when, ac.then),
        "",
        "Current UAT test case:",
        render_uat_candidate(target.title, target.description),
    ]
    if data.messages:
        parts += [
            "",
            "Reviewer feedback and additional context, most recent last:",
            render_conversation(ConversationRequest(messages=data.messages)),
        ]
    else:
        parts += [
            "",
            "No additional reviewer context was given; sharpen it for clarity and rigor.",
        ]
    return "\n".join(parts)


class UatCaseService:
    """Business logic for reviewing/regenerating UAT test cases.

    UAT cases are derived from persisted, already-accepted AC rows, not from
    chat content. `generate` runs the same agent roster as AC generation (see
    `run_uat_ensemble`), so each case has one version per model, the best
    shown and the rest swappable in. `regenerate_selected` stays single-model,
    like its AC counterpart. Everything is scored via the voting layer.
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        uat_repository: UatCaseRepository,
        ac_repository: AcceptanceCriteriaRepository,
        settings: Settings | None = None,
        regen_agent: Agent[None, GeneratedUatCase] | None = None,
        scorer: Scorer = score_candidates,
        uat_ensemble: UatEnsemble = run_uat_ensemble,
    ):
        self.sessions = session_repository
        self.uat = uat_repository
        self.ac = ac_repository
        self.settings = settings or default_settings
        # Built on first use so that a missing API key breaks regeneration
        # rather than every request that happens to construct this service.
        self._regen_agent = regen_agent
        self.scorer = scorer
        self.uat_ensemble = uat_ensemble

    @property
    def regen_agent(self) -> Agent[None, GeneratedUatCase]:
        if self._regen_agent is None:
            self._regen_agent = build_agent(
                self.settings, GeneratedUatCase, UAT_REGENERATION_SYSTEM_PROMPT
            )
        return self._regen_agent

    async def _cases_for(self, ac: AcceptanceCriterionRecord) -> list[UatCase]:
        """Every roster agent's test cases for one AC, scored, best version first.

        Raises GenerationError only when every agent failed for this AC.
        """
        prompt = render_ac_candidate(ac.title, ac.given, ac.when, ac.then)
        results, groups = await self.uat_ensemble(
            prompt, timeout_seconds=self.settings.generation_timeout_seconds
        )
        if not groups:
            reasons = "; ".join(f"{r.agent_id}: {r.error or 'no cases'}" for r in results)
            raise GenerationError(
                f"Every generation agent failed to write UAT cases for '{ac.title}'. {reasons}"
            )

        flat = [(group, candidate) for group in groups for candidate in group.candidates]
        texts = [render_uat_candidate(c.case.title, c.case.description) for _, c in flat]
        scores = await asyncio.to_thread(
            self.scorer, prompt=prompt, candidates=texts, settings=self.settings
        )
        for (_, candidate), score in zip(flat, scores, strict=True):
            candidate.relevance = score.relevance
            candidate.correctness = score.correctness
            candidate.understandability = score.understandability
            candidate.coverage = score.coverage
            candidate.overall_score = score.overall
            candidate.scoring_failed = score.failed

        cases: list[UatCase] = []
        for index, group in enumerate(groups):
            group.rank()
            winner = group.winner
            cases.append(
                UatCase(
                    id=-(index + 1),
                    ac_id=ac.id,
                    title=group.title,
                    description=winner.case.description,
                    scores=_scores_of(winner),
                    overall_score=winner.overall_score,
                    status="pending",
                    source_agent=winner.agent_id,
                    alternatives=[
                        UatCaseAlternative(
                            description=alt.case.description,
                            scores=_scores_of(alt),
                            overall_score=alt.overall_score,
                            source_agent=alt.agent_id,
                        )
                        for alt in group.alternatives
                    ],
                )
            )
        return cases

    def _build_group(
        self,
        ac_record: AcceptanceCriterionRecord,
        candidates: dict[tuple[int, int], list[UatCaseCandidateRecord]] | None = None,
    ) -> UatCaseGroup:
        if candidates is None:
            candidates = self.uat.candidates_by_case(ac_record.session_id)
        cases = self.uat.list_for_ac(ac_record.id)
        return UatCaseGroup(
            ac=_to_acceptance_criterion(ac_record),
            uat_cases=[_to_uat_case(c, candidates.get((c.ac_id, c.position))) for c in cases],
        )

    def _with_alternatives(self, record: UatCaseRecord) -> UatCase:
        candidates = self.uat.candidates_by_case(record.session_id)
        return _to_uat_case(record, candidates.get((record.ac_id, record.position)))

    def list_items(self, session_id: int) -> UatCaseGroupsResult | None:
        if self.sessions.get(session_id) is None:
            return None

        all_cases = self.uat.list_for_session(session_id)
        ac_ids = {c.ac_id for c in all_cases}
        ac_records = [self.ac.get(session_id, ac_id) for ac_id in ac_ids]
        present_records = [r for r in ac_records if r is not None]
        present_records.sort(key=lambda r: r.position)
        candidates = self.uat.candidates_by_case(session_id)
        groups = [self._build_group(r, candidates) for r in present_records]
        return UatCaseGroupsResult(groups=groups)

    async def generate(self, session_id: int) -> UatCaseGroupsResult | None:
        if self.sessions.get(session_id) is None:
            return None

        accepted = [r for r in self.ac.list_for_session(session_id) if r.status == "accepted"]
        # Every AC at once: each runs its own ensemble, so total time tracks
        # the slowest AC rather than the sum. Still all-or-nothing — a
        # GenerationError for any one AC aborts before persist_generated runs,
        # mirroring AC generation's replace-the-whole-batch semantics.
        cases = await asyncio.gather(*(self._cases_for(ac) for ac in accepted))
        self.uat.persist_generated(
            session_id, {ac.id: ac_cases for ac, ac_cases in zip(accepted, cases, strict=True)}
        )
        return self.list_items(session_id)

    def select_alternative(
        self, session_id: int, uat_id: int, candidate_id: int
    ) -> UatCase | None:
        """Swap another model's version of a test case in as the displayed one.

        None when the case or candidate doesn't exist, or the candidate belongs
        to a different case — the route maps all of those to 404.
        """
        record = self.uat.get(session_id, uat_id)
        candidate = self.uat.get_candidate(session_id, candidate_id)
        if record is None or candidate is None:
            return None
        if (candidate.ac_id, candidate.case_position) != (record.ac_id, record.position):
            return None
        if not candidate.is_winner:
            record = self.uat.swap_in_candidate(record, candidate)
        return self._with_alternatives(record)

    def update_text(
        self, session_id: int, uat_id: int, data: UatCaseTextUpdate
    ) -> UatCase | None:
        record = self.uat.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat.update_text(record, data.title, data.description)
        return self._with_alternatives(updated)

    def update_status(
        self, session_id: int, uat_id: int, data: UatCaseStatusUpdate
    ) -> UatCase | None:
        record = self.uat.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat.update_status(record, data.status)
        return self._with_alternatives(updated)

    def regenerate_selected(
        self, session_id: int, uat_id: int, data: UatRegenerateSelectedRequest
    ) -> UatRegenerateSelectedResponse | None:
        target = self.uat.get(session_id, uat_id)
        if target is None:
            return None
        ac = self.ac.get(session_id, target.ac_id)
        if ac is None:
            raise GenerationError(f"UAT case '{target.title}' has no parent acceptance criterion.")

        # Candidate id here is a throwaway placeholder — not persisted
        # until/unless the caller approves it via apply_approved.
        prompt = _build_regeneration_prompt(ac, target, data)
        try:
            result = self.regen_agent.run_sync(prompt)
        except Exception as error:
            raise GenerationError(
                f"{self.settings.llm_model} could not regenerate '{target.title}': {error}"
            ) from error

        case = result.output
        candidate_text = render_uat_candidate(case.title, case.description)
        [score] = self.scorer(prompt=prompt, candidates=[candidate_text], settings=self.settings)

        refined = UatCase(
            id=-1,
            ac_id=target.ac_id,
            title=case.title,
            description=case.description,
            scores=UatCaseScores(
                relevance=score.relevance,
                correctness=score.correctness,
                understandability=score.understandability,
                coverage=score.coverage,
            ),
            overall_score=score.overall,
            status="pending",
        )
        reply = ConversationMessage(
            role="assistant",
            text=f"Regenerated '{target.title}' using the additional context you provided.",
        )
        return UatRegenerateSelectedResponse(reply=reply, candidates=[refined])

    def apply_approved(
        self, session_id: int, uat_id: int, data: UatApplyApprovedRequest
    ) -> UatCaseGroup | None:
        if self.sessions.get(session_id) is None:
            return None
        target = self.uat.get(session_id, uat_id)
        if target is None:
            return None
        if not data.candidates:
            raise ValueError("Must approve at least one candidate; use Cancel to discard instead.")

        # Capture ac_id before replace_one deletes+commits the target row —
        # afterwards the `target` ORM object is expired and re-reading an
        # attribute off it would raise ObjectDeletedError.
        ac_id = target.ac_id
        self.uat.replace_one(session_id, ac_id, uat_id, data.candidates)
        ac_record = self.ac.get(session_id, ac_id)
        if ac_record is None:
            return None
        return self._build_group(ac_record)


def _scores_of(candidate) -> UatCaseScores:
    return UatCaseScores(
        relevance=candidate.relevance,
        correctness=candidate.correctness,
        understandability=candidate.understandability,
        coverage=candidate.coverage,
    )
