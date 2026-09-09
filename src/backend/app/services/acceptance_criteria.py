from pydantic_ai import Agent

from app.config import Settings, settings as default_settings
from app.models_acceptance_criteria import (
    AcceptanceCriterionRecord,
    AcceptanceCriterionStatusUpdate,
    AcceptanceCriterionTextUpdate,
    ApplyApprovedRequest,
    RegenerateSelectedRequest,
    RegenerateSelectedResponse,
)
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
    ConversationRequest,
)
from app.models_session import MessageRead
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.agents import GenerationError, build_agent
from app.services.conversation import GeneratedCriterion, render_conversation
from app.services.scoring import Scorer, render_ac_candidate, score_candidates

AC_REGENERATION_SYSTEM_PROMPT = """\
You are an expert business analyst refining ONE existing Gherkin acceptance
criterion (Given/When/Then) using a reviewer's follow-up feedback.

You will be given the criterion as it stands now, plus a conversation of
additional context or feedback from the reviewer. Produce exactly one
improved criterion covering the same underlying behaviour — do not invent an
unrelated new one. Keep it atomic, testable, and free of implementation
detail. If the reviewer's feedback conflicts with what's given, follow the
feedback; it is the more recent and more specific instruction.
"""


def _build_regeneration_prompt(
    target: AcceptanceCriterionRecord, data: RegenerateSelectedRequest
) -> str:
    parts = [
        "Current acceptance criterion:",
        render_ac_candidate(target.title, target.given, target.when, target.then),
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


class AcceptanceCriteriaService:
    """Business logic for reviewing/regenerating acceptance criteria.

    `regenerate_selected` is wired to a real model: it refines the target
    criterion using the reviewer's added context and returns exactly one
    candidate, scored via the voting layer (through `scorer`).
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        ac_repository: AcceptanceCriteriaRepository,
        message_repository: MessageRepository,
        uat_case_repository: UatCaseRepository,
        settings: Settings | None = None,
        regen_agent: Agent[None, GeneratedCriterion] | None = None,
        scorer: Scorer = score_candidates,
    ):
        self.sessions = session_repository
        self.ac = ac_repository
        self.messages = message_repository
        self.uat_cases = uat_case_repository
        self.settings = settings or default_settings
        # Built on first use so that a missing API key breaks regeneration
        # rather than every request that happens to construct this service.
        self._regen_agent = regen_agent
        self.scorer = scorer

    @property
    def regen_agent(self) -> Agent[None, GeneratedCriterion]:
        if self._regen_agent is None:
            self._regen_agent = build_agent(
                self.settings, GeneratedCriterion, AC_REGENERATION_SYSTEM_PROMPT
            )
        return self._regen_agent

    def list_items(self, session_id: int) -> list[AcceptanceCriterion] | None:
        if self.sessions.get(session_id) is None:
            return None
        return [_to_acceptance_criterion(r) for r in self.ac.list_for_session(session_id)]

    def update_text(
        self, session_id: int, ac_id: int, data: AcceptanceCriterionTextUpdate
    ) -> AcceptanceCriterion | None:
        record = self.ac.get(session_id, ac_id)
        if record is None:
            return None
        updated = self.ac.update_text(record, data.title, data.given, data.when, data.then)
        return _to_acceptance_criterion(updated)

    def update_status(
        self, session_id: int, ac_id: int, data: AcceptanceCriterionStatusUpdate
    ) -> AcceptanceCriterion | None:
        record = self.ac.get(session_id, ac_id)
        if record is None:
            return None
        updated = self.ac.update_status(record, data.status)
        return _to_acceptance_criterion(updated)

    def regenerate_selected(
        self, session_id: int, ac_id: int, data: RegenerateSelectedRequest
    ) -> RegenerateSelectedResponse | None:
        target = self.ac.get(session_id, ac_id)
        if target is None:
            return None

        # Candidate `id` here is a throwaway placeholder — it's not
        # persisted until/unless the caller approves it via apply_approved.
        prompt = _build_regeneration_prompt(target, data)
        try:
            result = self.regen_agent.run_sync(prompt)
        except Exception as error:
            raise GenerationError(
                f"{self.settings.llm_model} could not regenerate '{target.title}': {error}"
            ) from error

        criterion = result.output
        candidate_text = render_ac_candidate(
            criterion.title, criterion.given, criterion.when, criterion.then
        )
        [score] = self.scorer(prompt=prompt, candidates=[candidate_text], settings=self.settings)

        refined = AcceptanceCriterion(
            id=-1,
            title=criterion.title,
            given=criterion.given,
            when=criterion.when,
            then=criterion.then,
            scores=AcceptanceCriterionScores(
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
        return RegenerateSelectedResponse(reply=reply, candidates=[refined])

    def apply_approved(
        self, session_id: int, ac_id: int, data: ApplyApprovedRequest
    ) -> list[AcceptanceCriterion] | None:
        if self.sessions.get(session_id) is None:
            return None
        if self.ac.get(session_id, ac_id) is None:
            return None
        if not data.candidates:
            raise ValueError("Must approve at least one candidate; use Cancel to discard instead.")

        # The AC row being replaced may already have UAT cases generated
        # against it (via "Generate test cases..."). Those FK to this exact
        # ac_id, so they must be cleared before the row is deleted+replaced —
        # otherwise this would violate the FK in a database that enforces it.
        self.uat_cases.delete_for_ac(ac_id)
        updated = self.ac.replace_one(session_id, ac_id, data.candidates)
        return [_to_acceptance_criterion(r) for r in updated]

    def regenerate_all_kickoff(self, session_id: int) -> MessageRead | None:
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None

        rejected = self.ac.list_rejected(session_id)
        if rejected:
            titles = ", ".join(f"'{r.title}'" for r in rejected)
            text = (
                f"I've noted you rejected: {titles}. Could you share more context — "
                "any missing edge cases, wrong assumptions, or extra information — "
                "so I can improve the acceptance criteria on the next generation?"
            )
        else:
            text = (
                "Let's regenerate the acceptance criteria from scratch. Could you share "
                "any extra context that would help — missing edge cases, assumptions, "
                "or additional information?"
            )

        message = self.messages.create(
            session_id=session_id, role="assistant", text=text, attachments=[]
        )
        self.sessions.touch(chat_session)
        return MessageRead(
            id=message.id,
            role=message.role,
            text=message.text,
            attachments=[],
            created_at=message.created_at,
        )
