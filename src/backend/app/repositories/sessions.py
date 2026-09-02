from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models_session import ChatSession


class SessionRepository:
    """Owns all direct database access for ChatSession rows."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str) -> ChatSession:
        chat_session = ChatSession(name=name)
        self.session.add(chat_session)
        self.session.commit()
        self.session.refresh(chat_session)
        return chat_session

    def list_all(self) -> list[ChatSession]:
        statement = select(ChatSession).order_by(ChatSession.updated_at.desc())
        return list(self.session.exec(statement).all())

    def get(self, session_id: int) -> ChatSession | None:
        return self.session.get(ChatSession, session_id)

    def rename(self, chat_session: ChatSession, name: str) -> ChatSession:
        chat_session.name = name
        chat_session.updated_at = datetime.now(timezone.utc)
        self.session.add(chat_session)
        self.session.commit()
        self.session.refresh(chat_session)
        return chat_session

    def touch(self, chat_session: ChatSession) -> None:
        chat_session.updated_at = datetime.now(timezone.utc)
        self.session.add(chat_session)
        self.session.commit()

    def delete(self, chat_session: ChatSession) -> None:
        self.session.delete(chat_session)
        self.session.commit()
