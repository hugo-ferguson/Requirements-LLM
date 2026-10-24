"""Wire models for chat messages and generated acceptance criteria."""

from typing import Literal

from sqlmodel import Field, SQLModel


class ConversationAttachment(SQLModel):
    """A file's extracted text, sent along with a chat message."""

    filename: str
    content: str
    document_id: int | None = Field(
        default=None,
        description="ID of the ingested document, if uploaded via /documents/upload.",
    )


class ConversationMessage(SQLModel):
    """One chat turn, from either the user or the assistant."""

    role: Literal["user", "assistant"]
    text: str
    attachments: list[ConversationAttachment] = Field(default_factory=list)


class ConversationRequest(SQLModel):
    """A whole conversation, oldest message first."""

    messages: list[ConversationMessage]


class AcceptanceCriterionScores(SQLModel):
    """The voting layer's four rubric scores, each from 0 to 5."""

    relevance: float = Field(ge=0, le=5)
    correctness: float = Field(ge=0, le=5)
    understandability: float = Field(ge=0, le=5)
    coverage: float = Field(ge=0, le=5)


class AcceptanceCriterionAlternative(SQLModel):
    """Another model's version of the same criterion.

    `candidate_id` is set once stored, so the reviewer can swap it in.
    """

    candidate_id: int | None = None
    given: str
    when: str
    then: str
    scores: AcceptanceCriterionScores
    overall_score: float = Field(ge=0, le=5)
    source_agent: str | None = None


class AcceptanceCriterion(SQLModel):
    """An acceptance criterion as the frontend sees it."""

    id: int
    title: str
    given: str
    when: str
    then: str
    scores: AcceptanceCriterionScores
    overall_score: float = Field(ge=0, le=5)
    status: Literal["pending", "accepted", "rejected"] = "pending"

    # Defaults let older clients and apply-approved requests omit these.
    source_agent: str | None = None
    alternatives: list[AcceptanceCriterionAlternative] = Field(default_factory=list)
    uat_case_count: int = Field(
        default=0,
        description=(
            "Number of UAT cases generated from this AC, any status. Used to "
            "warn before an edit or regeneration affects them."
        ),
    )


class GenerateResult(SQLModel):
    """The ranked acceptance criteria for a session."""

    acceptance_criteria: list[AcceptanceCriterion]
