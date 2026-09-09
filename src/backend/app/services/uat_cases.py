from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.config import Settings, settings as default_settings
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


class GeneratedUatCase(BaseModel):
    """One UAT test case, as written by the model."""

    title: str = Field(description="Short label for the scenario under test")
    description: str = Field(
        description="Concrete steps/data a tester follows and the expected result"
    )


class GeneratedUatCases(BaseModel):
    cases: list[GeneratedUatCase] = Field(
        description="2-4 concrete UAT test cases grounded in the acceptance criterion",
        min_length=2,
        max_length=4,
    )


UAT_GENERATION_SYSTEM_PROMPT = """\
You are a QA engineer writing User Acceptance Test cases for one accepted
Gherkin acceptance criterion (Given/When/Then).

Write 2-4 concrete UAT test cases a human tester could execute to verify the
criterion: each needs a short title and a description with concrete
steps/data and the expected result. Cover the happy path and at least one
edge or negative case. Stay grounded in what the criterion actually states —
do not invent unrelated functionality.
"""

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


def _to_uat_case(record: UatCaseRecord) -> UatCase:
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
    chat content, so `generate`/`regenerate_selected` don't go through
    ConversationService — they use their own agents, scored via the voting
    layer (through `scorer`), same as AcceptanceCriteriaService.
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        uat_repository: UatCaseRepository,
        ac_repository: AcceptanceCriteriaRepository,
        settings: Settings | None = None,
        generation_agent: Agent[None, GeneratedUatCases] | None = None,
        regen_agent: Agent[None, GeneratedUatCase] | None = None,
        scorer: Scorer = score_candidates,
    ):
        self.sessions = session_repository
        self.uat = uat_repository
        self.ac = ac_repository
        self.settings = settings or default_settings
        # Built on first use so that a missing API key breaks generation
        # rather than every request that happens to construct this service.
        self._generation_agent = generation_agent
        self._regen_agent = regen_agent
        self.scorer = scorer

    @property
    def generation_agent(self) -> Agent[None, GeneratedUatCases]:
        if self._generation_agent is None:
            self._generation_agent = build_agent(
                self.settings, GeneratedUatCases, UAT_GENERATION_SYSTEM_PROMPT
            )
        return self._generation_agent

    @property
    def regen_agent(self) -> Agent[None, GeneratedUatCase]:
        if self._regen_agent is None:
            self._regen_agent = build_agent(
                self.settings, GeneratedUatCase, UAT_REGENERATION_SYSTEM_PROMPT
            )
        return self._regen_agent

    def _cases_for(self, ac: AcceptanceCriterionRecord) -> list[UatCase]:
        prompt = render_ac_candidate(ac.title, ac.given, ac.when, ac.then)
        try:
            result = self.generation_agent.run_sync(prompt)
        except Exception as error:
            raise GenerationError(
                f"{self.settings.llm_model} could not generate UAT cases for '{ac.title}': {error}"
            ) from error

        cases = result.output.cases
        candidate_texts = [render_uat_candidate(c.title, c.description) for c in cases]
        scores = self.scorer(prompt=prompt, candidates=candidate_texts, settings=self.settings)

        return [
            UatCase(
                id=-(index + 1),
                ac_id=ac.id,
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
            for index, (case, score) in enumerate(zip(cases, scores, strict=True))
        ]

    def _build_group(self, ac_record: AcceptanceCriterionRecord) -> UatCaseGroup:
        cases = self.uat.list_for_ac(ac_record.id)
        return UatCaseGroup(
            ac=_to_acceptance_criterion(ac_record),
            uat_cases=[_to_uat_case(c) for c in cases],
        )

    def list_items(self, session_id: int) -> UatCaseGroupsResult | None:
        if self.sessions.get(session_id) is None:
            return None

        all_cases = self.uat.list_for_session(session_id)
        ac_ids = {c.ac_id for c in all_cases}
        ac_records = [self.ac.get(session_id, ac_id) for ac_id in ac_ids]
        present_records = [r for r in ac_records if r is not None]
        present_records.sort(key=lambda r: r.position)
        groups = [self._build_group(r) for r in present_records]
        return UatCaseGroupsResult(groups=groups)

    def generate(self, session_id: int) -> UatCaseGroupsResult | None:
        if self.sessions.get(session_id) is None:
            return None

        accepted = [r for r in self.ac.list_for_session(session_id) if r.status == "accepted"]
        # All-or-nothing: a GenerationError on any one AC aborts before
        # persist_generated runs, mirroring AC generation's
        # replace-the-whole-batch semantics.
        cases_by_ac_id = {ac.id: self._cases_for(ac) for ac in accepted}
        self.uat.persist_generated(session_id, cases_by_ac_id)
        return self.list_items(session_id)

    def update_text(
        self, session_id: int, uat_id: int, data: UatCaseTextUpdate
    ) -> UatCase | None:
        record = self.uat.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat.update_text(record, data.title, data.description)
        return _to_uat_case(updated)

    def update_status(
        self, session_id: int, uat_id: int, data: UatCaseStatusUpdate
    ) -> UatCase | None:
        record = self.uat.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat.update_status(record, data.status)
        return _to_uat_case(updated)

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
