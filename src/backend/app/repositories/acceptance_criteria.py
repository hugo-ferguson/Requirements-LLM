from sqlmodel import Session, select

from app.models_acceptance_criteria import (
    AcceptanceCriterionCandidateRecord,
    AcceptanceCriterionRecord,
)
from app.models_conversation import AcceptanceCriterion


class AcceptanceCriteriaRepository:
    """Owns all direct database access for AcceptanceCriterionRecord rows."""

    def __init__(self, session: Session):
        self.session = session

    def persist_batch(
        self, session_id: int, items: list[AcceptanceCriterion]
    ) -> list[AcceptanceCriterionRecord]:
        """Replace the session's entire AC list with a freshly generated batch."""
        self.delete_for_session(session_id)
        records = [
            AcceptanceCriterionRecord(
                session_id=session_id,
                position=index,
                title=item.title,
                given=item.given,
                when=item.when,
                then=item.then,
                relevance=item.scores.relevance,
                correctness=item.scores.correctness,
                understandability=item.scores.understandability,
                coverage=item.scores.coverage,
                overall_score=item.overall_score,
                status="pending",
            )
            for index, item in enumerate(items)
        ]
        for record in records:
            self.session.add(record)

        self._persist_candidates(session_id, items)

        self.session.commit()
        for record in records:
            self.session.refresh(record)
        return records

    def _persist_candidates(
        self, session_id: int, items: list[AcceptanceCriterion]
    ) -> None:
        """Store every candidate behind each criterion, winner included.

        Written in the same transaction as the criteria themselves so the two
        can never disagree about what was generated. Criteria carrying no
        alternatives still get their winner row, which keeps provenance
        uniform whether or not title-anchored generation was used.
        """
        self.delete_candidates_for_session(session_id)

        for position, item in enumerate(items):
            rows = [
                AcceptanceCriterionCandidateRecord(
                    session_id=session_id,
                    criterion_position=position,
                    source_agent=item.source_agent or "unknown",
                    is_winner=True,
                    title=item.title,
                    given=item.given,
                    when=item.when,
                    then=item.then,
                    relevance=item.scores.relevance,
                    correctness=item.scores.correctness,
                    understandability=item.scores.understandability,
                    coverage=item.scores.coverage,
                    overall_score=item.overall_score,
                )
            ]
            rows.extend(
                AcceptanceCriterionCandidateRecord(
                    session_id=session_id,
                    criterion_position=position,
                    source_agent=alt.source_agent or "unknown",
                    is_winner=False,
                    title=item.title,
                    given=alt.given,
                    when=alt.when,
                    then=alt.then,
                    relevance=alt.scores.relevance,
                    correctness=alt.scores.correctness,
                    understandability=alt.scores.understandability,
                    coverage=alt.scores.coverage,
                    overall_score=alt.overall_score,
                )
                for alt in item.alternatives
            )
            for row in rows:
                self.session.add(row)

    def delete_candidates_for_session(self, session_id: int) -> None:
        statement = select(AcceptanceCriterionCandidateRecord).where(
            AcceptanceCriterionCandidateRecord.session_id == session_id
        )
        for row in self.session.exec(statement).all():
            self.session.delete(row)

    def candidates_for_session(
        self, session_id: int
    ) -> list[AcceptanceCriterionCandidateRecord]:
        """Every stored candidate for a session, grouped position then best first.

        Not yet read by any route — it exists so the alternatives are
        queryable for debugging and for the frontend work that follows.
        """
        statement = (
            select(AcceptanceCriterionCandidateRecord)
            .where(AcceptanceCriterionCandidateRecord.session_id == session_id)
            .order_by(
                AcceptanceCriterionCandidateRecord.criterion_position,
                AcceptanceCriterionCandidateRecord.overall_score.desc(),
            )
        )
        return list(self.session.exec(statement).all())

    def list_for_session(self, session_id: int) -> list[AcceptanceCriterionRecord]:
        statement = (
            select(AcceptanceCriterionRecord)
            .where(AcceptanceCriterionRecord.session_id == session_id)
            .order_by(AcceptanceCriterionRecord.position.asc())
        )
        return list(self.session.exec(statement).all())

    def get(self, session_id: int, ac_id: int) -> AcceptanceCriterionRecord | None:
        record = self.session.get(AcceptanceCriterionRecord, ac_id)
        if record is None or record.session_id != session_id:
            return None
        return record

    def list_rejected(self, session_id: int) -> list[AcceptanceCriterionRecord]:
        return [r for r in self.list_for_session(session_id) if r.status == "rejected"]

    def update_text(
        self, record: AcceptanceCriterionRecord, title: str, given: str, when: str, then: str
    ) -> AcceptanceCriterionRecord:
        record.title = title
        record.given = given
        record.when = when
        record.then = then
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def update_status(
        self, record: AcceptanceCriterionRecord, status: str
    ) -> AcceptanceCriterionRecord:
        record.status = status
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def replace_one(
        self, session_id: int, target_id: int, new_items: list[AcceptanceCriterion]
    ) -> list[AcceptanceCriterionRecord]:
        """Swap one persisted item for one or more new items, in place.

        The first new item reuses the freed position (so it visually replaces
        the original); any extra items are appended after the current list.
        Positions are renormalized to 0..N-1 afterwards so there are never gaps.
        """
        existing = self.list_for_session(session_id)
        target = next((r for r in existing if r.id == target_id), None)
        if target is None:
            return existing

        target_position = target.position
        remaining = [r for r in existing if r.id != target_id]
        self.session.delete(target)

        new_records = [
            AcceptanceCriterionRecord(
                session_id=session_id,
                position=0,  # renormalized below
                title=item.title,
                given=item.given,
                when=item.when,
                then=item.then,
                relevance=item.scores.relevance,
                correctness=item.scores.correctness,
                understandability=item.scores.understandability,
                coverage=item.scores.coverage,
                overall_score=item.overall_score,
                status="accepted",
            )
            for item in new_items
        ]

        merged = remaining[:target_position] + new_records + remaining[target_position:]
        for index, record in enumerate(merged):
            record.position = index
            self.session.add(record)
        self.session.commit()
        for record in merged:
            self.session.refresh(record)
        return merged

    def delete_for_session(self, session_id: int) -> None:
        # Candidates go with their criteria, always. Keeping this here rather
        # than in the caller means every delete path — session teardown and
        # the wholesale replace in persist_batch alike — cleans up both, and a
        # stale candidate row can never outlive the criterion it belonged to
        # and block the session from being deleted.
        self.delete_candidates_for_session(session_id)
        for record in self.list_for_session(session_id):
            self.session.delete(record)
        self.session.commit()
