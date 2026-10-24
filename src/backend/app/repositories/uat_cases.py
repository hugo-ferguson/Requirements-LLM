"""Database access for UAT test cases and their candidates."""

from sqlmodel import Session, col, func, select

from app.models_uat_cases import UatCase, UatCaseCandidateRecord, UatCaseRecord


class UatCaseRepository:
    """Reads and writes UAT cases and their candidates."""

    def __init__(self, session: Session):
        """Bind the repository to one database session."""
        self.session = session

    def list_for_session(self, session_id: int) -> list[UatCaseRecord]:
        """Return a session's cases, ordered by AC ID and then position."""
        statement = (
            select(UatCaseRecord)
            .where(UatCaseRecord.session_id == session_id)
            .order_by(col(UatCaseRecord.ac_id).asc(), col(UatCaseRecord.position).asc())
        )
        return list(self.session.exec(statement).all())

    def list_for_ac(self, ac_id: int) -> list[UatCaseRecord]:
        """Return one AC's cases in display order."""
        statement = (
            select(UatCaseRecord)
            .where(UatCaseRecord.ac_id == ac_id)
            .order_by(col(UatCaseRecord.position).asc())
        )
        return list(self.session.exec(statement).all())

    def counts_by_ac(self, session_id: int) -> dict[int, int]:
        """Return UAT case counts keyed by AC ID, leaving out ACs with none."""
        statement = (
            select(UatCaseRecord.ac_id, func.count())
            .where(UatCaseRecord.session_id == session_id)
            .group_by(UatCaseRecord.ac_id)
        )
        return {ac_id: count for ac_id, count in self.session.exec(statement).all()}

    def get(self, session_id: int, uat_id: int) -> UatCaseRecord | None:
        """Return a case, or None if it isn't in this session."""
        record = self.session.get(UatCaseRecord, uat_id)
        if record is None or record.session_id != session_id:
            return None
        return record

    def update_text(
        self, record: UatCaseRecord, title: str, description: str
    ) -> UatCaseRecord:
        """Overwrite a case's text with a reviewer's edit."""
        record.title = title
        record.description = description
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def update_status(self, record: UatCaseRecord, status: str) -> UatCaseRecord:
        """Set a case's review status."""
        record.status = status
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def replace_one(
        self,
        session_id: int,
        ac_id: int,
        target_id: int,
        replacements: list[UatCase],
    ) -> list[UatCaseRecord]:
        """Replace one test case with one or more new ones, in its place.

        Works like `AcceptanceCriteriaRepository.replace_one`, within one AC.

        Returns:
            The AC's cases in their new order, or unchanged if `target_id` isn't found.
        """
        existing = self.list_for_ac(ac_id)
        target = next((record for record in existing if record.id == target_id), None)
        if target is None:
            return existing

        target_position = target.position
        remaining = [record for record in existing if record.id != target_id]
        self.session.delete(target)

        new_records = [
            UatCaseRecord(
                session_id=session_id,
                ac_id=ac_id,
                position=0,  # Renumbered below.
                title=replacement.title,
                description=replacement.description,
                relevance=replacement.scores.relevance,
                correctness=replacement.scores.correctness,
                understandability=replacement.scores.understandability,
                coverage=replacement.scores.coverage,
                overall_score=replacement.overall_score,
                status="accepted",
            )
            for replacement in replacements
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

        # Candidates are keyed by position, so move them with the renumbering.
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

    def persist_batch(
        self, session_id: int, cases_by_ac_id: dict[int, list[UatCase]]
    ) -> None:
        """Replace a session's UAT cases and candidates with a new batch."""
        self.delete_for_session(session_id)
        for ac_id, cases in cases_by_ac_id.items():
            for position, case in enumerate(cases):
                record = UatCaseRecord(
                    session_id=session_id,
                    ac_id=ac_id,
                    position=position,
                    title=case.title,
                    description=case.description,
                    relevance=case.scores.relevance,
                    correctness=case.scores.correctness,
                    understandability=case.scores.understandability,
                    coverage=case.scores.coverage,
                    overall_score=case.overall_score,
                    status="pending",
                )
                self.session.add(record)
                self._add_candidates(session_id, ac_id, position, case)
        self.session.commit()

    def _add_candidates(
        self, session_id: int, ac_id: int, position: int, case: UatCase
    ) -> None:
        """Store every version of one case, the winner included."""
        self.session.add(
            UatCaseCandidateRecord(
                session_id=session_id,
                ac_id=ac_id,
                case_position=position,
                source_agent=case.source_agent or "unknown",
                is_winner=True,
                title=case.title,
                description=case.description,
                relevance=case.scores.relevance,
                correctness=case.scores.correctness,
                understandability=case.scores.understandability,
                coverage=case.scores.coverage,
                overall_score=case.overall_score,
            )
        )
        for alternative in case.alternatives:
            self.session.add(
                UatCaseCandidateRecord(
                    session_id=session_id,
                    ac_id=ac_id,
                    case_position=position,
                    source_agent=alternative.source_agent or "unknown",
                    is_winner=False,
                    title=case.title,
                    description=alternative.description,
                    relevance=alternative.scores.relevance,
                    correctness=alternative.scores.correctness,
                    understandability=alternative.scores.understandability,
                    coverage=alternative.scores.coverage,
                    overall_score=alternative.overall_score,
                )
            )

    def candidates_for_ac(self, ac_id: int) -> list[UatCaseCandidateRecord]:
        """Return one AC's candidates by case position, best first."""
        statement = (
            select(UatCaseCandidateRecord)
            .where(UatCaseCandidateRecord.ac_id == ac_id)
            .order_by(
                col(UatCaseCandidateRecord.case_position),
                col(UatCaseCandidateRecord.overall_score).desc(),
            )
        )
        return list(self.session.exec(statement).all())

    def candidates_by_case(
        self, session_id: int
    ) -> dict[tuple[int, int], list[UatCaseCandidateRecord]]:
        """Return a session's candidates keyed by (ac_id, position), best first."""
        statement = (
            select(UatCaseCandidateRecord)
            .where(UatCaseCandidateRecord.session_id == session_id)
            .order_by(col(UatCaseCandidateRecord.overall_score).desc())
        )
        grouped: dict[tuple[int, int], list[UatCaseCandidateRecord]] = {}
        for candidate in self.session.exec(statement).all():
            grouped.setdefault((candidate.ac_id, candidate.case_position), []).append(
                candidate
            )
        return grouped

    def get_candidate(
        self, session_id: int, candidate_id: int
    ) -> UatCaseCandidateRecord | None:
        """Return a candidate, or None if it isn't in this session."""
        candidate = self.session.get(UatCaseCandidateRecord, candidate_id)
        if candidate is None or candidate.session_id != session_id:
            return None
        return candidate

    def swap_in_candidate(
        self, record: UatCaseRecord, candidate: UatCaseCandidateRecord
    ) -> UatCaseRecord:
        """Make `candidate` the displayed version of `record`.

        The current text, manual edits included, becomes an alternative.
        Title and status don't change.
        """
        position_rows = [
            row
            for row in self.candidates_for_ac(record.ac_id)
            if row.case_position == record.position
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
        """Delete a session's cases and their candidates."""
        statement = select(UatCaseCandidateRecord).where(
            UatCaseCandidateRecord.session_id == session_id
        )
        for candidate in self.session.exec(statement).all():
            self.session.delete(candidate)
        for record in self.list_for_session(session_id):
            self.session.delete(record)
        self.session.commit()

    def delete_for_ac(self, ac_id: int) -> None:
        """Delete one AC's cases and their candidates."""
        for candidate in self.candidates_for_ac(ac_id):
            self.session.delete(candidate)
        for record in self.list_for_ac(ac_id):
            self.session.delete(record)
        self.session.commit()
