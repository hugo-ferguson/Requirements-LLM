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


class AcceptanceCriterionCandidateRecord(SQLModel, table=True):
    """Every agent's attempt at one criterion, winner and losers alike.

    Deliberately a separate table rather than extra columns on
    `AcceptanceCriterionRecord`. `create_db_and_tables` calls
    `SQLModel.metadata.create_all`, which creates missing *tables* but never
    adds columns to an existing one — so widening the criterion table would
    silently break every developer database already holding sessions, while a
    new table is created on next startup for everyone.

    The winner is stored here too, not just in `AcceptanceCriterionRecord`.
    Keeping the full candidate set in one place means provenance and scores
    are queryable uniformly, instead of the winner living under a different
    shape from the alternatives it beat.
    """

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    # Position of the criterion group this candidate belongs to, matching
    # `AcceptanceCriterionRecord.position`. Not a foreign key: criteria are
    # rewritten wholesale on regeneration, and a hard reference would either
    # block that or cascade deletes into the candidate history.
    criterion_position: int = Field(index=True)

    source_agent: str
    is_winner: bool = False

    title: str
    given: str
    when: str
    then: str

    relevance: float
    correctness: float
    understandability: float
    coverage: float
    overall_score: float

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
