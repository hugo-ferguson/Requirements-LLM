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
    """Convert a stored message for the chat and generation services."""
    return ConversationMessage(
        role=message.role,  # type: ignore[arg-type]
        text=message.text,
        attachments=[
            ConversationAttachment(**attachment) for attachment in message.attachments
        ],
    )


def _to_message_read(message: Message) -> MessageRead:
    """Convert a stored message for the API."""
    return MessageRead(
        id=message.id,  # type: ignore[arg-type]
        role=message.role,  # type: ignore[arg-type]
        text=message.text,
        attachments=[
            ConversationAttachment(**attachment) for attachment in message.attachments
        ],
        created_at=message.created_at,
    )


class SessionService:
    """Manage chat sessions and their messages, and generate ACs from them."""

    def __init__(
        self,
        session_repository: SessionRepository,
        message_repository: MessageRepository,
        conversation_service: ConversationService,
        ac_repository: AcceptanceCriteriaRepository,
        uat_case_repository: UatCaseRepository,
        generation_service: GenerationService,
    ):
        """Store the repositories and services a session touches."""
        self.session_repository = session_repository
        self.message_repository = message_repository
        self.conversation_service = conversation_service
        self.ac_repository = ac_repository
        self.uat_case_repository = uat_case_repository
        self.generation_service = generation_service

    def create_session(self, data: SessionCreate) -> ChatSession:
        """Create a session, named "New session" unless one is given."""
        return self.session_repository.create(data.name or "New session")

    def list_sessions(self) -> list[ChatSession]:
        """Return every session, most recently updated first."""
        return self.session_repository.list_all()

    def get_session_detail(self, session_id: int) -> SessionDetail | None:
        """Return a session and its messages, or None if it's missing."""
        chat_session = self.session_repository.get(session_id)
        if chat_session is None:
            return None
        history = self.message_repository.list_for_session(session_id)
        return SessionDetail(
            id=chat_session.id,  # type: ignore[arg-type]
            name=chat_session.name,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
            messages=[_to_message_read(message) for message in history],
        )

    def rename_session(
        self, session_id: int, data: SessionUpdate
    ) -> ChatSession | None:
        """Rename a session, or return None if it doesn't exist."""
        chat_session = self.session_repository.get(session_id)
        if chat_session is None:
            return None
        return self.session_repository.rename(chat_session, data.name)

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all its data; return False if it's missing."""
        chat_session = self.session_repository.get(session_id)
        if chat_session is None:
            return False
        self.message_repository.delete_for_session(session_id)
        # UAT cases reference their AC by foreign key, so delete them first.
        self.uat_case_repository.delete_for_session(session_id)
        self.ac_repository.delete_for_session(session_id)
        self.session_repository.delete(chat_session)
        return True

    def post_message(self, session_id: int, data: MessageCreate) -> MessageRead | None:
        """Store a user message and return the assistant's reply, or None if missing."""
        chat_session = self.session_repository.get(session_id)
        if chat_session is None:
            return None

        self.message_repository.create(
            session_id=session_id,
            role="user",
            text=data.text,
            attachments=[attachment.model_dump() for attachment in data.attachments],
        )

        history = self.message_repository.list_for_session(session_id)
        request = ConversationRequest(
            messages=[_to_conversation_message(message) for message in history]
        )
        reply = self.conversation_service.send_message(request)

        reply_message = self.message_repository.create(
            session_id=session_id, role="assistant", text=reply.text, attachments=[]
        )
        self.session_repository.touch(chat_session)
        return _to_message_read(reply_message)

    async def generate(self, session_id: int) -> GenerateResult | None:
        """Generate ACs from the conversation, replacing old ACs and UAT cases.

        Returns:
            The stored ACs, or None if the session is missing.

        Raises:
            NoUserStoryError: The conversation has no user story in it.
            GenerationError: Every generation agent failed.
        """
        chat_session = self.session_repository.get(session_id)
        if chat_session is None:
            return None

        history = self.message_repository.list_for_session(session_id)
        messages = [_to_conversation_message(message) for message in history]
        criteria = await self.generation_service.generate(messages)

        # UAT cases reference the old ACs by foreign key, so delete them first.
        self.uat_case_repository.delete_for_session(session_id)
        persisted = self.ac_repository.persist_batch(session_id, criteria)
        return GenerateResult(
            acceptance_criteria=to_acceptance_criteria(
                self.ac_repository, session_id, persisted
            )
        )
