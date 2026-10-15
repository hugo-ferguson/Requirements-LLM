"""Business logic for chat sessions, messages and AC generation."""

from app.models_conversation import (
    ConversationAttachment,
    ConversationMessage,
    ConversationRequest,
    GenerateResult,
)
from app.models_session import (
    ChatSession,
    Message,
    MessageCreate,
    MessageRead,
    SessionCreate,
    SessionDetail,
    SessionUpdate,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.acceptance_criteria import to_acceptance_criteria
from app.services.conversation import ConversationService
from app.services.generation import GenerationService


def _to_conversation_message(message: Message) -> ConversationMessage:
    # The shape the chat and generation services read.
    return ConversationMessage(
        role=message.role,  # type: ignore[arg-type]
        text=message.text,
        attachments=[ConversationAttachment(
            **attachment) for attachment in message.attachments],
    )


def _to_message_read(message: Message) -> MessageRead:
    # The shape the API returns.
    return MessageRead(
        id=message.id,  # type: ignore[arg-type]
        role=message.role,  # type: ignore[arg-type]
        text=message.text,
        attachments=[ConversationAttachment(
            **attachment) for attachment in message.attachments],
        created_at=message.created_at,
    )


class SessionService:
    """Business logic for sessions and messages, independent of HTTP."""

    def __init__(
        self,
        session_repository: SessionRepository,
        message_repository: MessageRepository,
        conversation_service: ConversationService,
        acceptance_criteria_repository: AcceptanceCriteriaRepository,
        uat_case_repository: UatCaseRepository,
        generation_service: GenerationService,
    ):
        """Wire up the repositories and services a session touches."""
        self.sessions = session_repository
        self.messages = message_repository
        self.conversation = conversation_service
        self.acceptance_criteria = acceptance_criteria_repository
        self.uat_cases = uat_case_repository
        self.generation = generation_service

    def create_session(self, data: SessionCreate) -> ChatSession:
        """Create a session, named "New session" unless one is given."""
        return self.sessions.create(data.name or "New session")

    def list_sessions(self) -> list[ChatSession]:
        """Return every session, most recently updated first."""
        return self.sessions.list_all()

    def get_session(self, session_id: int) -> ChatSession | None:
        """Return a session, or None if it doesn't exist."""
        return self.sessions.get(session_id)

    def get_session_detail(self, session_id: int) -> SessionDetail | None:
        """Return a session and its messages; None if it's missing."""
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None
        history = self.messages.list_for_session(session_id)
        return SessionDetail(
            id=chat_session.id,  # type: ignore[arg-type]
            name=chat_session.name,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
            messages=[_to_message_read(message) for message in history],
        )

    def rename_session(self, session_id: int, data: SessionUpdate) -> ChatSession | None:
        """Rename a session, or return None if it doesn't exist."""
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None
        return self.sessions.rename(chat_session, data.name)

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all its data; False if it's missing."""
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return False
        self.messages.delete_for_session(session_id)
        # UAT rows FK to acceptance_criteria rows, so they must go
        # first.
        self.uat_cases.delete_for_session(session_id)
        self.acceptance_criteria.delete_for_session(session_id)
        self.sessions.delete(chat_session)
        return True

    def post_message(self, session_id: int, data: MessageCreate) -> MessageRead | None:
        """Store a user message and return the assistant's stored reply.

        Return None when the session doesn't exist.
        """
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None

        self.messages.create(
            session_id=session_id,
            role="user",
            text=data.text,
            attachments=[attachment.model_dump()
                         for attachment in data.attachments],
        )

        history = self.messages.list_for_session(session_id)
        request = ConversationRequest(
            messages=[_to_conversation_message(m) for m in history])
        reply = self.conversation.send_message(request)

        reply_message = self.messages.create(
            session_id=session_id, role="assistant", text=reply.text, attachments=[]
        )
        self.sessions.touch(chat_session)
        return _to_message_read(reply_message)

    async def generate(self, session_id: int) -> GenerateResult | None:
        """Run the ensemble pipeline for a session and store the result.

        Async because the generation layer fans out to every enabled
        agent concurrently; see
        `app.generation.orchestrator.run_ensemble`.
        """
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None

        history = self.messages.list_for_session(session_id)
        messages = [_to_conversation_message(m) for m in history]
        criteria = await self.generation.generate(messages)

        # Regenerating ACs replaces their ids, so any UAT cases
        # generated against the old ids would otherwise be left as
        # orphaned rows (or violate the FK, in a database that enforces
        # it) — clear them too.
        self.uat_cases.delete_for_session(session_id)
        persisted = self.acceptance_criteria.persist_batch(
            session_id, criteria)
        return GenerateResult(
            acceptance_criteria=to_acceptance_criteria(
                self.acceptance_criteria, session_id, persisted
            )
        )
