from sqlmodel import Session, select

from app.models_uat_cases import UatCase, UatCaseRecord


class UatCaseRepository:
    """Owns all direct database access for UatCaseRecord rows."""

    def __init__(self, session: Session):
        self.session = session

    def list_for_session(self, session_id: int) -> list[UatCaseRecord]:
        statement = (
            select(UatCaseRecord)
            .where(UatCaseRecord.session_id == session_id)
            .order_by(UatCaseRecord.ac_id.asc(), UatCaseRecord.position.asc())
        )
        return list(self.session.exec(statement).all())

    def list_for_ac(self, ac_id: int) -> list[UatCaseRecord]:
        statement = (
            select(UatCaseRecord)
            .where(UatCaseRecord.ac_id == ac_id)
            .order_by(UatCaseRecord.position.asc())
        )
        return list(self.session.exec(statement).all())

    def get(self, session_id: int, uat_id: int) -> UatCaseRecord | None:
        record = self.session.get(UatCaseRecord, uat_id)
        if record is None or record.session_id != session_id:
            return None
        return record

    def update_text(self, record: UatCaseRecord, title: str, description: str) -> UatCaseRecord:
        record.title = title
        record.description = description
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def update_status(self, record: UatCaseRecord, status: str) -> UatCaseRecord:
        record.status = status
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def replace_one(
        self, session_id: int, ac_id: int, target_id: int, new_items: list[UatCase]
    ) -> list[UatCaseRecord]:
        """Swap one persisted UAT case for one or more new ones, in place,

        scoped entirely to the parent AC's own sublist — other ACs' UAT
        cases and positions are never touched. Mirrors
        AcceptanceCriteriaRepository.replace_one exactly, one level deeper.
        """
        existing = self.list_for_ac(ac_id)
        target = next((r for r in existing if r.id == target_id), None)
        if target is None:
            return existing

        target_position = target.position
        remaining = [r for r in existing if r.id != target_id]
        self.session.delete(target)

        new_records = [
            UatCaseRecord(
                session_id=session_id,
                ac_id=ac_id,
                position=0,  # renormalized below
                title=item.title,
                description=item.description,
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

    def persist_generated(self, session_id: int, cases_by_ac_id: dict[int, list[UatCase]]) -> None:
        """Replace the session's entire UAT batch with a freshly generated one."""
        self.delete_for_session(session_id)
        for ac_id, items in cases_by_ac_id.items():
            for index, item in enumerate(items):
                record = UatCaseRecord(
                    session_id=session_id,
                    ac_id=ac_id,
                    position=index,
                    title=item.title,
                    description=item.description,
                    relevance=item.scores.relevance,
                    correctness=item.scores.correctness,
                    understandability=item.scores.understandability,
                    coverage=item.scores.coverage,
                    overall_score=item.overall_score,
                    status="pending",
                )
                self.session.add(record)
        self.session.commit()

    def delete_for_session(self, session_id: int) -> None:
        for record in self.list_for_session(session_id):
            self.session.delete(record)
        self.session.commit()
