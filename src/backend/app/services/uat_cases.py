from app.models_acceptance_criteria import AcceptanceCriterionRecord
from app.models_conversation import AcceptanceCriterion, AcceptanceCriterionScores, ConversationMessage
from app.models_uat_cases import (
    UatApplyApprovedRequest,
    UatCase,
    UatCaseGroup,
    UatCaseGroupsResult,
    UatCaseRecord,
    UatCaseScores,
    UatCaseStatusUpdate,
    UatCaseTextUpdate,
    UatRegenerateSelectedRequest,
    UatRegenerateSelectedResponse,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository


def _to_acceptance_criterion(record: AcceptanceCriterionRecord) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        id=record.id,
        title=record.title,
        given=record.given,
        when=record.when,
        then=record.then,
        scores=AcceptanceCriterionScores(
            relevance=record.relevance,
            correctness=record.correctness,
            understandability=record.understandability,
            coverage=record.coverage,
        ),
        overall_score=record.overall_score,
        status=record.status,
    )


def _to_uat_case(record: UatCaseRecord) -> UatCase:
    return UatCase(
        id=record.id,
        ac_id=record.ac_id,
        title=record.title,
        description=record.description,
        scores=UatCaseScores(
            relevance=record.relevance,
            correctness=record.correctness,
            understandability=record.understandability,
            coverage=record.coverage,
        ),
        overall_score=record.overall_score,
        status=record.status,
    )


def _dummy_cases_for(ac: AcceptanceCriterionRecord) -> list[UatCase]:
    """Canned/dummy UAT cases for one AC. No AI is wired up yet (see
    ConversationService/AcceptanceCriteriaService) — this ignores anything
    about the AC beyond its title and returns fixed example data.
    """
    specs = [
        ("happy path", "Description for UAT 1", 9.2, 8.8, 9.4, 9.0, 9.1),
        ("edge case", "Description for UAT 2", 8.9, 8.6, 9.0, 8.7, 8.8),
        ("negative case", "Description for UAT 3", 8.5, 8.4, 8.7, 8.3, 8.5),
    ]
    return [
        UatCase(
            id=-(index + 1),
            ac_id=ac.id,
            title=f"{ac.title} — {label}",
            description=description,
            scores=UatCaseScores(
                relevance=relevance,
                correctness=correctness,
                understandability=understandability,
                coverage=coverage,
            ),
            overall_score=overall,
            status="pending",
        )
        for index, (label, description, relevance, correctness, understandability, coverage, overall) in enumerate(
            specs
        )
    ]


class UatCaseService:
    """Business logic for reviewing/regenerating UAT test cases.

    UAT cases are derived from persisted, already-accepted AC rows, not from
    chat content, so this never goes through ConversationService.
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        uat_repository: UatCaseRepository,
        ac_repository: AcceptanceCriteriaRepository,
    ):
        self.sessions = session_repository
        self.uat = uat_repository
        self.ac = ac_repository

    def _build_group(self, ac_record: AcceptanceCriterionRecord) -> UatCaseGroup:
        cases = self.uat.list_for_ac(ac_record.id)
        return UatCaseGroup(
            ac=_to_acceptance_criterion(ac_record),
            uat_cases=[_to_uat_case(c) for c in cases],
        )

    def list_items(self, session_id: int) -> UatCaseGroupsResult | None:
        if self.sessions.get(session_id) is None:
            return None

        all_cases = self.uat.list_for_session(session_id)
        ac_ids = {c.ac_id for c in all_cases}
        ac_records = [self.ac.get(session_id, ac_id) for ac_id in ac_ids]
        present_records = [r for r in ac_records if r is not None]
        present_records.sort(key=lambda r: r.position)
        groups = [self._build_group(r) for r in present_records]
        return UatCaseGroupsResult(groups=groups)

    def generate(self, session_id: int) -> UatCaseGroupsResult | None:
        if self.sessions.get(session_id) is None:
            return None

        accepted = [r for r in self.ac.list_for_session(session_id) if r.status == "accepted"]
        cases_by_ac_id = {ac.id: _dummy_cases_for(ac) for ac in accepted}
        self.uat.persist_generated(session_id, cases_by_ac_id)
        return self.list_items(session_id)

    def update_text(
        self, session_id: int, uat_id: int, data: UatCaseTextUpdate
    ) -> UatCase | None:
        record = self.uat.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat.update_text(record, data.title, data.description)
        return _to_uat_case(updated)

    def update_status(
        self, session_id: int, uat_id: int, data: UatCaseStatusUpdate
    ) -> UatCase | None:
        record = self.uat.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat.update_status(record, data.status)
        return _to_uat_case(updated)

    def regenerate_selected(
        self, session_id: int, uat_id: int, data: UatRegenerateSelectedRequest
    ) -> UatRegenerateSelectedResponse | None:
        target = self.uat.get(session_id, uat_id)
        if target is None:
            return None

        # Canned/dummy regeneration: ignores `data.messages` entirely, mirroring
        # AcceptanceCriteriaService.regenerate_selected's convention. Candidate
        # ids here are throwaway placeholders — not persisted until approved.
        refined = UatCase(
            id=-1,
            ac_id=target.ac_id,
            title=target.title,
            description=f"{target.description} (refined with more detail)",
            scores=UatCaseScores(relevance=9.4, correctness=9.1, understandability=9.3, coverage=9.0),
            overall_score=9.2,
            status="pending",
        )
        additional = UatCase(
            id=-2,
            ac_id=target.ac_id,
            title=f"{target.title} — additional scenario",
            description="Description for an additional UAT case covering a related scenario",
            scores=UatCaseScores(relevance=8.7, correctness=8.5, understandability=8.8, coverage=8.3),
            overall_score=8.6,
            status="pending",
        )

        reply = ConversationMessage(
            role="assistant",
            text=f"Regenerated '{target.title}' with more detail, and added an additional UAT case.",
        )
        return UatRegenerateSelectedResponse(reply=reply, candidates=[refined, additional])

    def apply_approved(
        self, session_id: int, uat_id: int, data: UatApplyApprovedRequest
    ) -> UatCaseGroup | None:
        if self.sessions.get(session_id) is None:
            return None
        target = self.uat.get(session_id, uat_id)
        if target is None:
            return None
        if not data.candidates:
            raise ValueError("Must approve at least one candidate; use Cancel to discard instead.")

        # Capture ac_id before replace_one deletes+commits the target row —
        # afterwards the `target` ORM object is expired and re-reading an
        # attribute off it would raise ObjectDeletedError.
        ac_id = target.ac_id
        self.uat.replace_one(session_id, ac_id, uat_id, data.candidates)
        ac_record = self.ac.get(session_id, ac_id)
        if ac_record is None:
            return None
        return self._build_group(ac_record)
