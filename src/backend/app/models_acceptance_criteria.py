from datetime import datetime, timezone
from typing import Literal

from sqlmodel import Field, SQLModel

from app.models_conversation import AcceptanceCriterion, ConversationMessage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AcceptanceCriterionRecord(SQLModel, table=True):
    """A persisted acceptance criterion belonging to a ChatSession.

    Named `AcceptanceCriterionRecord` (not `AcceptanceCriterion`) to avoid
    clashing with the wire schema of that name in `models_conversation.py`,
    which stays the single canonical "AC as seen by the frontend" shape.
    """

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    position: int

    title: str
    given: str
    when: str
    then: str

    relevance: float
    correctness: float
    understandability: float
    coverage: float
    overall_score: float

    status: str = "pending"
    created_at: datetime = Field(default_factory=_utcnow)


class AcceptanceCriterionTextUpdate(SQLModel):
    title: str
    given: str
    when: str
    then: str


class AcceptanceCriterionStatusUpdate(SQLModel):
    status: Literal["pending", "accepted", "rejected"]


class RegenerateSelectedRequest(SQLModel):
    messages: list[ConversationMessage]


class RegenerateSelectedResponse(SQLModel):
    reply: ConversationMessage
    candidates: list[AcceptanceCriterion]


class ApplyApprovedRequest(SQLModel):
    candidates: list[AcceptanceCriterion]
