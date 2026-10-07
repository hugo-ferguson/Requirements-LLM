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
    relevance: float = Field(ge=0, le=5)
    correctness: float = Field(ge=0, le=5)
    understandability: float = Field(ge=0, le=5)
    coverage: float = Field(ge=0, le=5)


class AcceptanceCriterionAlternative(SQLModel):
    """A losing candidate for the same title as its parent criterion.

    Carries no `status`: an alternative is not a reviewable item in its own
    right, it is the answer to "what did the other model say?". It does carry
    `candidate_id` once persisted, so the reviewer can swap it in.
    """

    candidate_id: int | None = None
    given: str
    when: str
    then: str
    scores: AcceptanceCriterionScores
    overall_score: float = Field(ge=0, le=5)
    source_agent: str | None = None


class AcceptanceCriterion(SQLModel):
    id: int
    title: str
    given: str
    when: str
    then: str
    scores: AcceptanceCriterionScores
    overall_score: float = Field(ge=0, le=5)
    status: Literal["pending", "accepted", "rejected"] = "pending"

    # Both fields are additive with defaults, so an existing client that knows
    # nothing about them keeps working unchanged — the frontend can adopt the
    # alternatives UI whenever it is ready rather than in lockstep with this.
    source_agent: str | None = None
    alternatives: list[AcceptanceCriterionAlternative] = Field(default_factory=list)


class GenerateResult(SQLModel):
    acceptance_criteria: list[AcceptanceCriterion]
