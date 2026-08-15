from app.models_acceptance_criteria import (
    AcceptanceCriterionRecord,
    AcceptanceCriterionStatusUpdate,
    AcceptanceCriterionTextUpdate,
    ApplyApprovedRequest,
    RegenerateSelectedRequest,
    RegenerateSelectedResponse,
)
from app.models_conversation import AcceptanceCriterion, AcceptanceCriterionScores, ConversationMessage
from app.models_session import MessageRead
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository


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

    No AI is wired up here either (see `ConversationService`): regeneration
    ignores the chat content and returns fixed example candidates.
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        ac_repository: AcceptanceCriteriaRepository,
        message_repository: MessageRepository,
        uat_case_repository: UatCaseRepository,
    ):
        self.sessions = session_repository
        self.ac = ac_repository
        self.messages = message_repository
        self.uat_cases = uat_case_repository

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

        # Canned/dummy regeneration: ignores `data.messages` entirely, mirroring
        # ConversationService's "no AI wired up yet" convention. Candidate `id`s
        # here are throwaway placeholders — they're not persisted until/unless
        # the caller approves them via apply_approved.
        refined = AcceptanceCriterion(
            id=-1,
            title=target.title,
            given=f"{target.given}, where a valid credential means a correct username and password combination",
            when=target.when,
            then=target.then,
            scores=AcceptanceCriterionScores(
                relevance=9.5, correctness=9.2, understandability=9.4, coverage=9.0
            ),
            overall_score=9.3,
            status="pending",
        )
        new_addition = AcceptanceCriterion(
            id=-2,
            title="Database entry created on registration",
            given="a new user completes registration with valid details",
            when="the registration form is submitted",
            then="a corresponding record is created in the users database table",
            scores=AcceptanceCriterionScores(
                relevance=8.8, correctness=8.6, understandability=8.9, coverage=8.4
            ),
            overall_score=8.7,
            status="pending",
        )

        reply = ConversationMessage(
            role="assistant",
            text=(
                f"Regenerated '{target.title}' with more detail on valid credentials, "
                "and added a new acceptance criterion about database entries."
            ),
        )
        return RegenerateSelectedResponse(reply=reply, candidates=[refined, new_addition])

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
