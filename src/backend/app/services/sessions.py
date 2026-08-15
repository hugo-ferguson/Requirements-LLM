from app.models_acceptance_criteria import AcceptanceCriterionRecord
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
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
from app.services.conversation import ConversationService


def _to_conversation_message(message: Message) -> ConversationMessage:
    return ConversationMessage(
        role=message.role,
        text=message.text,
        attachments=[ConversationAttachment(**attachment) for attachment in message.attachments],
    )


def _to_message_read(message: Message) -> MessageRead:
    return MessageRead(
        id=message.id,
        role=message.role,
        text=message.text,
        attachments=[ConversationAttachment(**attachment) for attachment in message.attachments],
        created_at=message.created_at,
    )


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


class SessionService:
    """Business logic for sessions and their messages, independent of HTTP concerns."""

    def __init__(
        self,
        session_repository: SessionRepository,
        message_repository: MessageRepository,
        conversation_service: ConversationService,
        acceptance_criteria_repository: AcceptanceCriteriaRepository,
    ):
        self.sessions = session_repository
        self.messages = message_repository
        self.conversation = conversation_service
        self.acceptance_criteria = acceptance_criteria_repository

    def create_session(self, data: SessionCreate) -> ChatSession:
        return self.sessions.create(data.name or "New session")

    def list_sessions(self) -> list[ChatSession]:
        return self.sessions.list_all()

    def get_session(self, session_id: int) -> ChatSession | None:
        return self.sessions.get(session_id)

    def get_session_detail(self, session_id: int) -> SessionDetail | None:
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None
        history = self.messages.list_for_session(session_id)
        return SessionDetail(
            id=chat_session.id,
            name=chat_session.name,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
            messages=[_to_message_read(message) for message in history],
        )

    def rename_session(self, session_id: int, data: SessionUpdate) -> ChatSession | None:
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None
        return self.sessions.rename(chat_session, data.name)

    def delete_session(self, session_id: int) -> bool:
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return False
        self.messages.delete_for_session(session_id)
        self.acceptance_criteria.delete_for_session(session_id)
        self.sessions.delete(chat_session)
        return True

    def post_message(self, session_id: int, data: MessageCreate) -> MessageRead | None:
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None

        self.messages.create(
            session_id=session_id,
            role="user",
            text=data.text,
            attachments=[attachment.model_dump() for attachment in data.attachments],
        )

        history = self.messages.list_for_session(session_id)
        request = ConversationRequest(messages=[_to_conversation_message(m) for m in history])
        reply = self.conversation.send_message(request)

        reply_message = self.messages.create(
            session_id=session_id, role="assistant", text=reply.text, attachments=[]
        )
        self.sessions.touch(chat_session)
        return _to_message_read(reply_message)

    def generate(self, session_id: int) -> GenerateResult | None:
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None

        history = self.messages.list_for_session(session_id)
        request = ConversationRequest(messages=[_to_conversation_message(m) for m in history])
        reply = self.conversation.generate(request)

        persisted = self.acceptance_criteria.persist_batch(session_id, reply.acceptance_criteria)
        return GenerateResult(acceptance_criteria=[_to_acceptance_criterion(r) for r in persisted])
