from typing import Literal

from sqlmodel import Field, SQLModel


class ConversationAttachment(SQLModel):
    filename: str
    content: str
    # Set when the attachment came from /documents/upload, so the message can
    # be traced back to the ingested (and embedded) document.
    document_id: int | None = None


class ConversationMessage(SQLModel):
    role: Literal["user", "assistant"]
    text: str
    attachments: list[ConversationAttachment] = Field(default_factory=list)


class ConversationRequest(SQLModel):
    messages: list[ConversationMessage]


class AcceptanceCriterionScores(SQLModel):
    relevance: float = Field(ge=0, le=10)
    correctness: float = Field(ge=0, le=10)
    understandability: float = Field(ge=0, le=10)
    coverage: float = Field(ge=0, le=10)


class AcceptanceCriterion(SQLModel):
    id: int
    title: str
    given: str
    when: str
    then: str
    scores: AcceptanceCriterionScores
    overall_score: float = Field(ge=0, le=10)
    status: Literal["pending", "accepted", "rejected"] = "pending"


class GenerateResult(SQLModel):
    acceptance_criteria: list[AcceptanceCriterion]
