"""Database access for chat messages."""

from sqlmodel import Session, col, select

from app.models_session import Message


class MessageRepository:
    """Reads and writes `Message` rows."""

    def __init__(self, session: Session):
        """Bind the repository to one database session."""
        self.session = session

    def create(
        self, session_id: int, role: str, text: str, attachments: list[dict]
    ) -> Message:
        """Store a message in a session and return it."""
        message = Message(
            session_id=session_id, role=role, text=text, attachments=attachments
        )
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        return message

    def list_for_session(self, session_id: int) -> list[Message]:
        """Return a session's messages, oldest first."""
        statement = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(col(Message.id).asc())
        )
        return list(self.session.exec(statement).all())

    def delete_for_session(self, session_id: int) -> None:
        """Delete every message in a session."""
        for message in self.list_for_session(session_id):
            self.session.delete(message)
        self.session.commit()
