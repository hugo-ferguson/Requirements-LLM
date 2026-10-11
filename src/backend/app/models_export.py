from datetime import datetime

from sqlmodel import SQLModel

from app.models_conversation import AcceptanceCriterionScores
from app.models_uat_cases import UatCaseScores


class ExportUatCase(SQLModel):
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
    """What the downloaded JSON file contains."""

    session_name: str
    exported_at: datetime
    acceptance_criteria: list[ExportAcceptanceCriterion]


class ExportWarnings(SQLModel):
    """What the export leaves out, so the user can confirm before downloading."""

    # ACs that are pending or rejected, and so aren't exported.
    unapproved_ac_count: int
    # Titles of approved ACs that have no approved UAT case to nest.
    approved_ac_without_uat: list[str]


class ExportResponse(SQLModel):
    document: ExportDocument
    warnings: ExportWarnings
