"""Business logic for exporting a session's approved requirements as JSON."""

from datetime import UTC, datetime

from app.models_export import (
    ExportAcceptanceCriterion,
    ExportDocument,
    ExportResponse,
    ExportUatCase,
    ExportWarnings,
)
from app.models_uat_cases import UatCaseRecord, UatCaseScores
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.acceptance_criteria import to_acceptance_criteria


class ExportService:
    """Builds the downloadable export: approved ACs with their approved UAT cases."""

    def __init__(
        self,
        session_repository: SessionRepository,
        ac_repository: AcceptanceCriteriaRepository,
        uat_case_repository: UatCaseRepository,
    ):
        """Store the repositories the export reads from."""
        self.session_repository = session_repository
        self.ac_repository = ac_repository
        self.uat_case_repository = uat_case_repository

    def build(self, session_id: int) -> ExportResponse | None:
        """Build a session's export, or return None if the session is missing."""
        chat_session = self.session_repository.get(session_id)
        if chat_session is None:
            return None

        records = self.ac_repository.list_for_session(session_id)
        approved = [record for record in records if record.status == "accepted"]

        approved_cases_by_ac: dict[int, list[UatCaseRecord]] = {}
        for case in self.uat_case_repository.list_for_session(session_id):
            if case.status == "accepted":
                approved_cases_by_ac.setdefault(case.ac_id, []).append(case)
        uat_candidates = self.uat_case_repository.candidates_by_case(session_id)

        def uat_source_agent(case: UatCaseRecord) -> str | None:
            """Return the agent that wrote the case's displayed version."""
            candidates = uat_candidates.get((case.ac_id, case.position), [])
            return next(
                (
                    candidate.source_agent
                    for candidate in candidates
                    if candidate.is_winner
                ),
                None,
            )

        # to_acceptance_criteria already resolves each AC's source agent.
        criteria: list[ExportAcceptanceCriterion] = []
        for criterion in to_acceptance_criteria(
            self.ac_repository, session_id, approved
        ):
            criteria.append(
                ExportAcceptanceCriterion(
                    title=criterion.title,
                    given=criterion.given,
                    when=criterion.when,
                    then=criterion.then,
                    scores=criterion.scores,
                    overall_score=criterion.overall_score,
                    source_agent=criterion.source_agent,
                    uat_cases=[
                        ExportUatCase(
                            title=case.title,
                            description=case.description,
                            scores=UatCaseScores(
                                relevance=case.relevance,
                                correctness=case.correctness,
                                understandability=case.understandability,
                                coverage=case.coverage,
                            ),
                            overall_score=case.overall_score,
                            source_agent=uat_source_agent(case),
                        )
                        for case in approved_cases_by_ac.get(criterion.id, [])
                    ],
                )
            )

        return ExportResponse(
            document=ExportDocument(
                session_name=chat_session.name,
                exported_at=datetime.now(UTC),
                acceptance_criteria=criteria,
            ),
            warnings=ExportWarnings(
                unapproved_ac_count=len(records) - len(approved),
                approved_ac_without_uat=[
                    criterion.title for criterion in criteria if not criterion.uat_cases
                ],
            ),
        )
