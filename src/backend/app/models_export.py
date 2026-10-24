"""API models for the JSON export of a session's approved requirements."""

from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models_conversation import AcceptanceCriterionScores
from app.models_uat_cases import UatCaseScores


class ExportUatCase(SQLModel):
    """An approved UAT case, as it appears in the export."""

    title: str
    description: str
    scores: UatCaseScores
    overall_score: float
    source_agent: str | None = None


class ExportAcceptanceCriterion(SQLModel):
    """An approved AC, with only its approved UAT cases nested under it."""

    title: str
    given: str
    when: str
    then: str
    scores: AcceptanceCriterionScores
    overall_score: float
    source_agent: str | None = None
    uat_cases: list[ExportUatCase]


class ExportDocument(SQLModel):
    """The contents of the downloaded JSON file."""

    session_name: str
    exported_at: datetime
    acceptance_criteria: list[ExportAcceptanceCriterion]


class ExportWarnings(SQLModel):
    """What the export leaves out, shown before downloading."""

    unapproved_ac_count: int = Field(
        description="Number of pending or rejected ACs, which aren't exported."
    )
    approved_ac_without_uat: list[str] = Field(
        description="Titles of approved ACs that have no approved UAT case."
    )


class ExportResponse(SQLModel):
    """The export document and the warnings about what it leaves out."""

    document: ExportDocument
    warnings: ExportWarnings
