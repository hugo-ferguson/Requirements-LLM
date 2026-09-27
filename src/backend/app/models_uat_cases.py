from datetime import datetime, timezone
from typing import Literal

from sqlmodel import Field, SQLModel

from app.models_conversation import AcceptanceCriterion, ConversationMessage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UatCaseRecord(SQLModel, table=True):
    """A persisted UAT test case belonging to a parent AcceptanceCriterionRecord.

    Named `UatCaseRecord` (not `UatCase`) to avoid clashing with the wire
    schema of that name below, same rationale as `AcceptanceCriterionRecord`.
    """

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    ac_id: int = Field(foreign_key="acceptancecriterionrecord.id", index=True)
    position: int  # scoped per ac_id, not per session

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
    """Every agent's version of one UAT case, winner and alternatives alike.

    The UAT counterpart of `AcceptanceCriterionCandidateRecord`, keyed the same
    way: by the case's parent AC and its position within that AC's list. No
    foreign key, for the same reason — cases are rewritten wholesale, and a
    hard reference would block that or cascade into the candidate history.
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
    relevance: float = Field(ge=0, le=5)
    correctness: float = Field(ge=0, le=5)
    understandability: float = Field(ge=0, le=5)
    coverage: float = Field(ge=0, le=5)


class UatCaseAlternative(SQLModel):
    """Another model's version of the same test case; swappable via `candidate_id`."""

    candidate_id: int | None = None
    description: str
    scores: UatCaseScores
    overall_score: float = Field(ge=0, le=5)
    source_agent: str | None = None


class UatCase(SQLModel):
    id: int
    ac_id: int
    title: str
    description: str
    scores: UatCaseScores
    overall_score: float = Field(ge=0, le=5)
    status: Literal["pending", "accepted", "rejected"] = "pending"
    # Additive with defaults, like the AC equivalents, so older clients keep working.
    source_agent: str | None = None
    alternatives: list[UatCaseAlternative] = Field(default_factory=list)


class UatCaseGroup(SQLModel):
    ac: AcceptanceCriterion
    uat_cases: list[UatCase]


class UatCaseGroupsResult(SQLModel):
    groups: list[UatCaseGroup]


class UatCaseTextUpdate(SQLModel):
    title: str
    description: str


class UatCaseStatusUpdate(SQLModel):
    status: Literal["pending", "accepted", "rejected"]


class UatRegenerateSelectedRequest(SQLModel):
    messages: list[ConversationMessage]


class UatRegenerateSelectedResponse(SQLModel):
    reply: ConversationMessage
    candidates: list[UatCase]


class UatApplyApprovedRequest(SQLModel):
    candidates: list[UatCase]
