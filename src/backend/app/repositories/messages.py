from sqlmodel import Session, select

from app.models_session import Message


class MessageRepository:
    """Owns all direct database access for Message rows."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, session_id: int, role: str, text: str, attachments: list[dict]) -> Message:
        message = Message(session_id=session_id, role=role, text=text, attachments=attachments)
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        return message

    def list_for_session(self, session_id: int) -> list[Message]:
        statement = (
            select(Message).where(Message.session_id == session_id).order_by(Message.id.asc())
        )
        return list(self.session.exec(statement).all())

    def delete_for_session(self, session_id: int) -> None:
        for message in self.list_for_session(session_id):
            self.session.delete(message)
        self.session.commit()
