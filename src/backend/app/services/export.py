from datetime import datetime, timezone

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
        self.sessions = session_repository
        self.ac = ac_repository
        self.uat_cases = uat_case_repository

    def build(self, session_id: int) -> ExportResponse | None:
        chat_session = self.sessions.get(session_id)
        if chat_session is None:
            return None

        records = self.ac.list_for_session(session_id)
        approved = [r for r in records if r.status == "accepted"]

        approved_cases: dict[int, list[UatCaseRecord]] = {}
        for case in self.uat_cases.list_for_session(session_id):
            if case.status == "accepted":
                approved_cases.setdefault(case.ac_id, []).append(case)
        uat_candidates = self.uat_cases.candidates_by_case(session_id)

        def uat_source_agent(case: UatCaseRecord) -> str | None:
            candidates = uat_candidates.get((case.ac_id, case.position), [])
            return next((c.source_agent for c in candidates if c.is_winner), None)

        # The wire conversion already resolves each AC's source_agent.
        criteria = []
        for criterion in to_acceptance_criteria(self.ac, session_id, approved):
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
                        for case in approved_cases.get(criterion.id, [])
                    ],
                )
            )

        return ExportResponse(
            document=ExportDocument(
                session_name=chat_session.name,
                exported_at=datetime.now(timezone.utc),
                acceptance_criteria=criteria,
            ),
            warnings=ExportWarnings(
                unapproved_ac_count=len(records) - len(approved),
                approved_ac_without_uat=[c.title for c in criteria if not c.uat_cases],
            ),
        )
