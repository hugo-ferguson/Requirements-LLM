"""Database access for acceptance criteria and their candidates."""

from sqlmodel import Session, col, select

from app.models_acceptance_criteria import (
    AcceptanceCriterionCandidateRecord,
    AcceptanceCriterionRecord,
)
from app.models_conversation import AcceptanceCriterion


class AcceptanceCriteriaRepository:
    """Reads and writes acceptance criteria and their candidates."""

    def __init__(self, session: Session):
        """Bind the repository to one database session."""
        self.session = session

    def persist_batch(
        self, session_id: int, criteria: list[AcceptanceCriterion]
    ) -> list[AcceptanceCriterionRecord]:
        """Replace a session's ACs and candidates with a new batch, kept in order."""
        self.delete_for_session(session_id)
        records = [
            AcceptanceCriterionRecord(
                session_id=session_id,
                position=index,
                title=criterion.title,
                given=criterion.given,
                when=criterion.when,
                then=criterion.then,
                relevance=criterion.scores.relevance,
                correctness=criterion.scores.correctness,
                understandability=criterion.scores.understandability,
                coverage=criterion.scores.coverage,
                overall_score=criterion.overall_score,
                status="pending",
            )
            for index, criterion in enumerate(criteria)
        ]
        for record in records:
            self.session.add(record)

        self._persist_candidates(session_id, criteria)

        self.session.commit()
        for record in records:
            self.session.refresh(record)
        return records

    def _persist_candidates(
        self, session_id: int, criteria: list[AcceptanceCriterion]
    ) -> None:
        """Store every version of each criterion, the winner included.

        Doesn't commit. Every criterion gets a winner row to record its source agent.
        """
        self.delete_candidates_for_session(session_id)

        for position, criterion in enumerate(criteria):
            rows = [
                AcceptanceCriterionCandidateRecord(
                    session_id=session_id,
                    criterion_position=position,
                    source_agent=criterion.source_agent or "unknown",
                    is_winner=True,
                    title=criterion.title,
                    given=criterion.given,
                    when=criterion.when,
                    then=criterion.then,
                    relevance=criterion.scores.relevance,
                    correctness=criterion.scores.correctness,
                    understandability=criterion.scores.understandability,
                    coverage=criterion.scores.coverage,
                    overall_score=criterion.overall_score,
                )
            ]
            rows.extend(
                AcceptanceCriterionCandidateRecord(
                    session_id=session_id,
                    criterion_position=position,
                    source_agent=alternative.source_agent or "unknown",
                    is_winner=False,
                    title=criterion.title,
                    given=alternative.given,
                    when=alternative.when,
                    then=alternative.then,
                    relevance=alternative.scores.relevance,
                    correctness=alternative.scores.correctness,
                    understandability=alternative.scores.understandability,
                    coverage=alternative.scores.coverage,
                    overall_score=alternative.overall_score,
                )
                for alternative in criterion.alternatives
            )
            for row in rows:
                self.session.add(row)

    def delete_candidates_for_session(self, session_id: int) -> None:
        """Delete every stored candidate in a session."""
        statement = select(AcceptanceCriterionCandidateRecord).where(
            AcceptanceCriterionCandidateRecord.session_id == session_id
        )
        for row in self.session.exec(statement).all():
            self.session.delete(row)

    def candidates_for_session(
        self, session_id: int
    ) -> list[AcceptanceCriterionCandidateRecord]:
        """Return a session's candidates by position, best first."""
        statement = (
            select(AcceptanceCriterionCandidateRecord)
            .where(AcceptanceCriterionCandidateRecord.session_id == session_id)
            .order_by(
                col(AcceptanceCriterionCandidateRecord.criterion_position),
                col(AcceptanceCriterionCandidateRecord.overall_score).desc(),
            )
        )
        return list(self.session.exec(statement).all())

    def candidates_by_position(
        self, session_id: int
    ) -> dict[int, list[AcceptanceCriterionCandidateRecord]]:
        """Return a session's candidates keyed by criterion position, best first."""
        grouped: dict[int, list[AcceptanceCriterionCandidateRecord]] = {}
        for candidate in self.candidates_for_session(session_id):
            grouped.setdefault(candidate.criterion_position, []).append(candidate)
        return grouped

    def get_candidate(
        self, session_id: int, candidate_id: int
    ) -> AcceptanceCriterionCandidateRecord | None:
        """Return a candidate, or None if it isn't in this session."""
        candidate = self.session.get(AcceptanceCriterionCandidateRecord, candidate_id)
        if candidate is None or candidate.session_id != session_id:
            return None
        return candidate

    def swap_in_candidate(
        self,
        record: AcceptanceCriterionRecord,
        candidate: AcceptanceCriterionCandidateRecord,
    ) -> AcceptanceCriterionRecord:
        """Make `candidate` the displayed version of `record`.

        The current text, manual edits included, becomes an alternative.
        Title and status don't change.
        """
        position_rows = self.candidates_by_position(record.session_id).get(
            record.position, []
        )
        outgoing = next((row for row in position_rows if row.is_winner), None)
        if outgoing is None:
            # An AC from regenerate-selected has no candidate rows of its own.
            outgoing = AcceptanceCriterionCandidateRecord(
                session_id=record.session_id,
                criterion_position=record.position,
                source_agent="unknown",
                title=record.title,
                given=record.given,
                when=record.when,
                then=record.then,
                relevance=record.relevance,
                correctness=record.correctness,
                understandability=record.understandability,
                coverage=record.coverage,
                overall_score=record.overall_score,
            )
        outgoing.given = record.given
        outgoing.when = record.when
        outgoing.then = record.then
        outgoing.is_winner = False
        self.session.add(outgoing)

        record.given = candidate.given
        record.when = candidate.when
        record.then = candidate.then
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

    def list_for_session(self, session_id: int) -> list[AcceptanceCriterionRecord]:
        """Return a session's criteria in display order."""
        statement = (
            select(AcceptanceCriterionRecord)
            .where(AcceptanceCriterionRecord.session_id == session_id)
            .order_by(col(AcceptanceCriterionRecord.position).asc())
        )
        return list(self.session.exec(statement).all())

    def get(self, session_id: int, ac_id: int) -> AcceptanceCriterionRecord | None:
        """Return a criterion, or None if it isn't in this session."""
        record = self.session.get(AcceptanceCriterionRecord, ac_id)
        if record is None or record.session_id != session_id:
            return None
        return record

    def list_rejected(self, session_id: int) -> list[AcceptanceCriterionRecord]:
        """Return the criteria a reviewer has rejected."""
        return [
            record
            for record in self.list_for_session(session_id)
            if record.status == "rejected"
        ]

    def update_text(
        self,
        record: AcceptanceCriterionRecord,
        title: str,
        given: str,
        when: str,
        then: str,
    ) -> AcceptanceCriterionRecord:
        """Overwrite a criterion's text with a reviewer's edit."""
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
        """Set a criterion's review status."""
        record.status = status
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def replace_one(
        self,
        session_id: int,
        target_id: int,
        replacements: list[AcceptanceCriterion],
    ) -> list[AcceptanceCriterionRecord]:
        """Replace one criterion with one or more new ones, in its place.

        Positions are renumbered from 0. Replacements are stored as accepted,
        and the replaced criterion's candidates are deleted.

        Returns:
            The criteria in their new order, or unchanged if `target_id` isn't found.
        """
        existing = self.list_for_session(session_id)
        target = next((record for record in existing if record.id == target_id), None)
        if target is None:
            return existing

        target_position = target.position
        remaining = [record for record in existing if record.id != target_id]
        self.session.delete(target)

        new_records = [
            AcceptanceCriterionRecord(
                session_id=session_id,
                position=0,  # Renumbered below.
                title=replacement.title,
                given=replacement.given,
                when=replacement.when,
                then=replacement.then,
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
        for candidate in self.candidates_for_session(session_id):
            new_position = new_position_by_old.get(candidate.criterion_position)
            if new_position is None:
                self.session.delete(candidate)
            else:
                candidate.criterion_position = new_position
                self.session.add(candidate)
        self.session.commit()
        for record in merged:
            self.session.refresh(record)
        return merged

    def delete_for_session(self, session_id: int) -> None:
        """Delete a session's criteria and their candidates."""
        # Done here for every caller, as leftover candidates block session deletion.
        self.delete_candidates_for_session(session_id)
        for record in self.list_for_session(session_id):
            self.session.delete(record)
        self.session.commit()
