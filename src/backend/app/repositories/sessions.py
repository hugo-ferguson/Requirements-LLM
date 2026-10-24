"""Database access for chat sessions."""

from datetime import UTC, datetime

from sqlmodel import Session, col, select

from app.models_session import ChatSession


class SessionRepository:
    """Reads and writes `ChatSession` rows."""

    def __init__(self, session: Session):
        """Bind the repository to one database session."""
        self.session = session

    def create(self, name: str) -> ChatSession:
        """Create a session with the given name and return it."""
        chat_session = ChatSession(name=name)
        self.session.add(chat_session)
        self.session.commit()
        self.session.refresh(chat_session)
        return chat_session

    def list_all(self) -> list[ChatSession]:
        """Return every session, most recently updated first."""
        statement = select(ChatSession).order_by(col(ChatSession.updated_at).desc())
        return list(self.session.exec(statement).all())

    def get(self, session_id: int) -> ChatSession | None:
        """Return a session by ID, or None if it doesn't exist."""
        return self.session.get(ChatSession, session_id)

    def rename(self, chat_session: ChatSession, name: str) -> ChatSession:
        """Rename a session and mark it as updated."""
        chat_session.name = name
        chat_session.updated_at = datetime.now(UTC)
        self.session.add(chat_session)
        self.session.commit()
        self.session.refresh(chat_session)
        return chat_session

    def touch(self, chat_session: ChatSession) -> None:
        """Mark a session as updated now, moving it to the top of the list."""
        chat_session.updated_at = datetime.now(UTC)
        self.session.add(chat_session)
        self.session.commit()

    def delete(self, chat_session: ChatSession) -> None:
        """Delete a session row."""
        self.session.delete(chat_session)
        self.session.commit()
