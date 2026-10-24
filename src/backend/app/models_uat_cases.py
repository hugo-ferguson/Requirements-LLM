"""Table and API models for UAT test cases and their candidates."""

from datetime import UTC, datetime
from typing import Literal

from sqlmodel import Field, SQLModel

from app.models_conversation import AcceptanceCriterion, ConversationMessage


def _utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class UatCaseRecord(SQLModel, table=True):
    """A stored UAT test case under one AC. `UatCase` is the API shape."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    ac_id: int = Field(foreign_key="acceptancecriterionrecord.id", index=True)
    position: int  # Within the parent AC, not the session.

    title: str
    description: str

    relevance: float
    correctness: float
    understandability: float
    coverage: float
    overall_score: float

    status: str = "pending"
    created_at: datetime = Field(default_factory=_utcnow)


class UatCaseCandidateRecord(SQLModel, table=True):
    """Every agent's version of one test case, the winner included.

    Keyed by parent AC and case position. `ac_id` isn't a foreign key, as
    cascading deletes would wipe the history or block regeneration.
    """

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    ac_id: int = Field(index=True)
    case_position: int

    source_agent: str
    is_winner: bool = False

    title: str
    description: str

    relevance: float
    correctness: float
    understandability: float
    coverage: float
    overall_score: float

    created_at: datetime = Field(default_factory=_utcnow)


class UatCaseScores(SQLModel):
    """The voting layer's four rubric scores, each from 0 to 5."""

    relevance: float = Field(ge=0, le=5)
    correctness: float = Field(ge=0, le=5)
    understandability: float = Field(ge=0, le=5)
    coverage: float = Field(ge=0, le=5)


class UatCaseAlternative(SQLModel):
    """Another model's version of the same test case.

    `candidate_id` is set once stored, so the reviewer can swap it in.
    """

    candidate_id: int | None = None
    description: str
    scores: UatCaseScores
    overall_score: float = Field(ge=0, le=5)
    source_agent: str | None = None


class UatCase(SQLModel):
    """A UAT test case as the frontend sees it."""

    id: int
    ac_id: int
    title: str
    description: str
    scores: UatCaseScores
    overall_score: float = Field(ge=0, le=5)
    status: Literal["pending", "accepted", "rejected"] = "pending"
    # Defaults let older clients and apply-approved requests omit these.
    source_agent: str | None = None
    alternatives: list[UatCaseAlternative] = Field(default_factory=list)


class UatCaseGroup(SQLModel):
    """An AC together with the test cases generated from it."""

    ac: AcceptanceCriterion
    uat_cases: list[UatCase]


class UatCaseGroupsResult(SQLModel):
    """A session's test cases, grouped by AC."""

    groups: list[UatCaseGroup]


class UatCaseTextUpdate(SQLModel):
    """Request body for a reviewer's manual edit of a test case."""

    title: str
    description: str


class UatCaseStatusUpdate(SQLModel):
    """Request body for accepting, rejecting or resetting a test case."""

    status: Literal["pending", "accepted", "rejected"]


class UatRegenerateSelectedRequest(SQLModel):
    """The conversation that steers regenerating one test case."""

    messages: list[ConversationMessage]


class UatRegenerateSelectedResponse(SQLModel):
    """The assistant's reply and the unsaved replacement candidates."""

    reply: ConversationMessage
    candidates: list[UatCase]


class UatApplyApprovedRequest(SQLModel):
    """The candidates a reviewer approved to replace a test case."""

    candidates: list[UatCase]
