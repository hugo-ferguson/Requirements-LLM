"""Table and API models for acceptance criteria and their candidates."""

from datetime import UTC, datetime
from typing import Literal

from sqlmodel import Field, SQLModel

from app.models_conversation import AcceptanceCriterion, ConversationMessage


def _utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class AcceptanceCriterionRecord(SQLModel, table=True):
    """A stored acceptance criterion. `AcceptanceCriterion` is the API shape."""

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
    """Every agent's version of one criterion, the winner included.

    A separate table because startup creates new tables but never adds columns.
    """

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    # Relates to `AcceptanceCriterionRecord.position`. Not a foreign key, as
    # cascading deletes would wipe the history or block regeneration.
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
    """Request body for a reviewer's manual edit of an AC."""

    title: str
    given: str
    when: str
    then: str


class AcceptanceCriterionStatusUpdate(SQLModel):
    """Request body for accepting, rejecting or resetting an AC."""

    status: Literal["pending", "accepted", "rejected"]


class RegenerateSelectedRequest(SQLModel):
    """The conversation that steers regenerating one AC."""

    messages: list[ConversationMessage]


class RegenerateSelectedResponse(SQLModel):
    """The assistant's reply and the unsaved replacement candidates."""

    reply: ConversationMessage
    candidates: list[AcceptanceCriterion]


class ApplyApprovedRequest(SQLModel):
    """The candidates a reviewer approved to replace an AC."""

    candidates: list[AcceptanceCriterion]


class SelectAlternativeResponse(SQLModel):
    """The AC after swapping in another model's version."""

    acceptance_criterion: AcceptanceCriterion
    uat_cases_affected: int = Field(
        description="Number of UAT cases written against the old wording."
    )
