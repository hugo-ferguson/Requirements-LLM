from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from app.models_conversation import ConversationAttachment


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatSession(SQLModel, table=True):
    """A persisted chat session, analogous to a single ChatGPT conversation.

    Named `ChatSession` (not `Session`) to avoid clashing with `sqlmodel.Session`,
    which is already used throughout the app as the DB session type.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = "New session"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Message(SQLModel, table=True):
    """A single persisted message within a ChatSession."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    role: str
    text: str
    attachments: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class SessionCreate(SQLModel):
    name: str | None = None


class SessionUpdate(SQLModel):
    name: str


class SessionSummary(SQLModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime


class MessageCreate(SQLModel):
    text: str
    attachments: list[ConversationAttachment] = Field(default_factory=list)


class MessageRead(SQLModel):
    id: int
    role: Literal["user", "assistant"]
    text: str
    attachments: list[ConversationAttachment]
    created_at: datetime


class SessionDetail(SessionSummary):
    messages: list[MessageRead]
