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


class UatCaseScores(SQLModel):
    relevance: float = Field(ge=0, le=10)
    correctness: float = Field(ge=0, le=10)
    understandability: float = Field(ge=0, le=10)
    coverage: float = Field(ge=0, le=10)


class UatCase(SQLModel):
    id: int
    ac_id: int
    title: str
    description: str
    scores: UatCaseScores
    overall_score: float = Field(ge=0, le=10)
    status: Literal["pending", "accepted", "rejected"] = "pending"


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
