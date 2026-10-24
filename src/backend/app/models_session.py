"""Table and API models for chat sessions and their messages."""

from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from app.models_conversation import ConversationAttachment


def _utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class ChatSession(SQLModel, table=True):
    """A stored chat session. Named to avoid clashing with `sqlmodel.Session`."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = "New session"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Message(SQLModel, table=True):
    """A single stored message in a chat session."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    role: str
    text: str
    attachments: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class SessionCreate(SQLModel):
    """Request body for creating a session; the name is optional."""

    name: str | None = None


class SessionUpdate(SQLModel):
    """Request body for renaming a session."""

    name: str


class SessionSummary(SQLModel):
    """A session without its messages, as listed in the sidebar."""

    id: int
    name: str
    created_at: datetime
    updated_at: datetime


class MessageCreate(SQLModel):
    """Request body for a new user message."""

    text: str
    attachments: list[ConversationAttachment] = Field(default_factory=list)


class MessageRead(SQLModel):
    """A stored message as returned by the API."""

    id: int
    role: Literal["user", "assistant"]
    text: str
    attachments: list[ConversationAttachment]
    created_at: datetime


class SessionDetail(SessionSummary):
    """A session together with its full message history."""

    messages: list[MessageRead]
