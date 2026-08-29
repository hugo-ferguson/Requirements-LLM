from sqlmodel import Session, select

from app.models_acceptance_criteria import AcceptanceCriterionRecord
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
        self.session.commit()
        for record in records:
            self.session.refresh(record)
        return records

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
        for record in self.list_for_session(session_id):
            self.session.delete(record)
        self.session.commit()
