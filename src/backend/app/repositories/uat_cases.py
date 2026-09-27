from sqlmodel import Session, select

from app.models_uat_cases import UatCase, UatCaseCandidateRecord, UatCaseRecord


class UatCaseRepository:
    """Owns all direct database access for UatCaseRecord rows and their candidates."""

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
        new_position_by_old = {
            record.position: index
            for index, record in enumerate(merged)
            if record.id is not None
        }
        for index, record in enumerate(merged):
            record.position = index
            self.session.add(record)

        # Candidates are keyed by position, so they follow the renumbering or
        # alternatives end up on a neighbouring case. The replaced case's own
        # candidates go with it.
        for candidate in self.candidates_for_ac(ac_id):
            new_position = new_position_by_old.get(candidate.case_position)
            if new_position is None:
                self.session.delete(candidate)
            else:
                candidate.case_position = new_position
                self.session.add(candidate)
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
                self._add_candidates(session_id, ac_id, index, item)
        self.session.commit()

    def _add_candidates(self, session_id: int, ac_id: int, position: int, item: UatCase) -> None:
        """Store the winner and every alternative behind one case."""
        self.session.add(
            UatCaseCandidateRecord(
                session_id=session_id,
                ac_id=ac_id,
                case_position=position,
                source_agent=item.source_agent or "unknown",
                is_winner=True,
                title=item.title,
                description=item.description,
                relevance=item.scores.relevance,
                correctness=item.scores.correctness,
                understandability=item.scores.understandability,
                coverage=item.scores.coverage,
                overall_score=item.overall_score,
            )
        )
        for alt in item.alternatives:
            self.session.add(
                UatCaseCandidateRecord(
                    session_id=session_id,
                    ac_id=ac_id,
                    case_position=position,
                    source_agent=alt.source_agent or "unknown",
                    is_winner=False,
                    title=item.title,
                    description=alt.description,
                    relevance=alt.scores.relevance,
                    correctness=alt.scores.correctness,
                    understandability=alt.scores.understandability,
                    coverage=alt.scores.coverage,
                    overall_score=alt.overall_score,
                )
            )

    def candidates_for_ac(self, ac_id: int) -> list[UatCaseCandidateRecord]:
        statement = (
            select(UatCaseCandidateRecord)
            .where(UatCaseCandidateRecord.ac_id == ac_id)
            .order_by(
                UatCaseCandidateRecord.case_position,
                UatCaseCandidateRecord.overall_score.desc(),
            )
        )
        return list(self.session.exec(statement).all())

    def candidates_by_case(
        self, session_id: int
    ) -> dict[tuple[int, int], list[UatCaseCandidateRecord]]:
        """Every stored candidate for a session, keyed by (ac_id, case position)."""
        statement = (
            select(UatCaseCandidateRecord)
            .where(UatCaseCandidateRecord.session_id == session_id)
            .order_by(UatCaseCandidateRecord.overall_score.desc())
        )
        grouped: dict[tuple[int, int], list[UatCaseCandidateRecord]] = {}
        for candidate in self.session.exec(statement).all():
            grouped.setdefault((candidate.ac_id, candidate.case_position), []).append(candidate)
        return grouped

    def get_candidate(self, session_id: int, candidate_id: int) -> UatCaseCandidateRecord | None:
        candidate = self.session.get(UatCaseCandidateRecord, candidate_id)
        if candidate is None or candidate.session_id != session_id:
            return None
        return candidate

    def swap_in_candidate(
        self, record: UatCaseRecord, candidate: UatCaseCandidateRecord
    ) -> UatCaseRecord:
        """Make `candidate` the displayed version of `record`, demoting the old one.

        Same rules as the AC swap: the outgoing version is saved from the
        record's current text, so a manual edit becomes a swappable
        alternative; title and status are left alone.
        """
        position_rows = [
            row for row in self.candidates_for_ac(record.ac_id) if row.case_position == record.position
        ]
        outgoing = next((row for row in position_rows if row.is_winner), None)
        if outgoing is None:
            # A case from regenerate-selected has no candidate rows of its own.
            outgoing = UatCaseCandidateRecord(
                session_id=record.session_id,
                ac_id=record.ac_id,
                case_position=record.position,
                source_agent="unknown",
                title=record.title,
                description=record.description,
                relevance=record.relevance,
                correctness=record.correctness,
                understandability=record.understandability,
                coverage=record.coverage,
                overall_score=record.overall_score,
            )
        outgoing.description = record.description
        outgoing.is_winner = False
        self.session.add(outgoing)

        record.description = candidate.description
        record.relevance = candidate.relevance
        record.correctness = candidate.correctness
        record.understandability = candidate.understandability
        record.coverage = candidate.coverage
        record.overall_score = candidate.overall_score
        candidate.is_winner = True
        self.session.add(candidate)
        self.session.add(record)

        self.session.commit()
        self.session.refresh(record)
        return record

    def delete_for_session(self, session_id: int) -> None:
        statement = select(UatCaseCandidateRecord).where(
            UatCaseCandidateRecord.session_id == session_id
        )
        for candidate in self.session.exec(statement).all():
            self.session.delete(candidate)
        for record in self.list_for_session(session_id):
            self.session.delete(record)
        self.session.commit()

    def delete_for_ac(self, ac_id: int) -> None:
        for candidate in self.candidates_for_ac(ac_id):
            self.session.delete(candidate)
        for record in self.list_for_ac(ac_id):
            self.session.delete(record)
        self.session.commit()
