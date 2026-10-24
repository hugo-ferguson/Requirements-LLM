"""Business logic for reviewing and regenerating acceptance criteria."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.config import Settings
from app.config import settings as default_settings
from app.llm_config import get_models_config
from app.models_acceptance_criteria import (
    AcceptanceCriterionCandidateRecord,
    AcceptanceCriterionRecord,
    AcceptanceCriterionStatusUpdate,
    AcceptanceCriterionTextUpdate,
    ApplyApprovedRequest,
    RegenerateSelectedRequest,
    RegenerateSelectedResponse,
    SelectAlternativeResponse,
)
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionAlternative,
    AcceptanceCriterionScores,
    ConversationMessage,
    ConversationRequest,
)
from app.models_session import MessageRead
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.agents import GenerationError, build_agent
from app.services.conversation import render_conversation
from app.services.scoring import Scorer, render_ac_candidate, score_candidates

AC_REGENERATION_SYSTEM_PROMPT = """\
You are an expert business analyst refining ONE existing Gherkin acceptance
criterion (Given/When/Then) using a reviewer's follow-up feedback.

You will be given the criterion as it stands now, plus a conversation of
additional context or feedback from the reviewer. Produce exactly one
improved criterion covering the same underlying behaviour — do not invent an
unrelated new one. Keep it atomic, testable, and free of implementation
detail. If the reviewer's feedback conflicts with what's given, follow the
feedback; it is the more recent and more specific instruction.
"""


class GeneratedCriterion(BaseModel):
    """One acceptance criterion, as written by the model."""

    title: str = Field(
        description="Short label for the behaviour, e.g. 'Successful login → /home'"
    )
    given: str = Field(description="Precondition or initial system/user state")
    when: str = Field(description="The specific action or event being tested")
    then: str = Field(description="The expected, observable outcome")


def _build_regeneration_prompt(
    target: AcceptanceCriterionRecord, data: RegenerateSelectedRequest
) -> str:
    """Build the prompt from the current criterion and any reviewer feedback."""
    parts = [
        "Current acceptance criterion:",
        render_ac_candidate(target.title, target.given, target.when, target.then),
    ]
    if data.messages:
        parts += [
            "",
            "Reviewer feedback and additional context, most recent last:",
            render_conversation(ConversationRequest(messages=data.messages)),
        ]
    else:
        parts += [
            "",
            "No additional reviewer context was given; sharpen it for clarity and "
            "rigor.",
        ]
    return "\n".join(parts)


def to_acceptance_criterion(
    record: AcceptanceCriterionRecord,
    candidates: list[AcceptanceCriterionCandidateRecord] | None = None,
    uat_case_count: int = 0,
) -> AcceptanceCriterion:
    """Build the wire model for one AC, with its alternatives attached.

    Args:
        record: The stored criterion.
        candidates: Candidates at its position. The winner gives `source_agent`.
        uat_case_count: Number of UAT cases generated from this AC.
    """
    candidates = candidates or []
    winner = next((candidate for candidate in candidates if candidate.is_winner), None)
    return AcceptanceCriterion(
        id=record.id,  # type: ignore[arg-type]
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
        status=record.status,  # type: ignore[arg-type]
        source_agent=winner.source_agent if winner else None,
        alternatives=[
            AcceptanceCriterionAlternative(
                candidate_id=candidate.id,
                given=candidate.given,
                when=candidate.when,
                then=candidate.then,
                scores=AcceptanceCriterionScores(
                    relevance=candidate.relevance,
                    correctness=candidate.correctness,
                    understandability=candidate.understandability,
                    coverage=candidate.coverage,
                ),
                overall_score=candidate.overall_score,
                source_agent=candidate.source_agent,
            )
            for candidate in candidates
            if not candidate.is_winner
        ],
        uat_case_count=uat_case_count,
    )


def to_acceptance_criteria(
    ac_repository: AcceptanceCriteriaRepository,
    session_id: int,
    records: list[AcceptanceCriterionRecord],
    uat_case_counts: dict[int, int] | None = None,
) -> list[AcceptanceCriterion]:
    """Build wire models for several ACs, in order, with one candidate query.

    Args:
        ac_repository: Source of the candidates.
        session_id: The records' session.
        records: Stored criteria, in output order.
        uat_case_counts: UAT case counts by AC ID. Missing ACs get 0.
    """
    candidates_by_position = ac_repository.candidates_by_position(session_id)
    uat_case_counts = uat_case_counts or {}
    return [
        to_acceptance_criterion(
            record,
            candidates_by_position.get(record.position),
            uat_case_counts.get(record.id, 0),
        )
        for record in records
    ]


class AcceptanceCriteriaService:
    """Review and regenerate a session's acceptance criteria."""

    def __init__(
        self,
        session_repository: SessionRepository,
        ac_repository: AcceptanceCriteriaRepository,
        message_repository: MessageRepository,
        uat_case_repository: UatCaseRepository,
        settings: Settings | None = None,
        regeneration_agent: Agent[None, GeneratedCriterion] | None = None,
        scorer: Scorer = score_candidates,
    ):
        """Store the dependencies. Tests can pass their own agent and scorer."""
        self.session_repository = session_repository
        self.ac_repository = ac_repository
        self.message_repository = message_repository
        self.uat_case_repository = uat_case_repository
        self.settings = settings or default_settings
        # Built on first use, so a missing API key only breaks regeneration.
        self._regeneration_agent = regeneration_agent
        self.scorer = scorer

    @property
    def regeneration_agent(self) -> Agent[None, GeneratedCriterion]:
        """The regeneration agent, built on first use."""
        if self._regeneration_agent is None:
            self._regeneration_agent = build_agent(
                GeneratedCriterion, AC_REGENERATION_SYSTEM_PROMPT
            )
        return self._regeneration_agent

    def list_for_session(self, session_id: int) -> list[AcceptanceCriterion] | None:
        """Return a session's ACs, or None if the session is missing."""
        if self.session_repository.get(session_id) is None:
            return None
        return to_acceptance_criteria(
            self.ac_repository,
            session_id,
            self.ac_repository.list_for_session(session_id),
            self.uat_case_repository.counts_by_ac(session_id),
        )

    def update_text(
        self, session_id: int, ac_id: int, data: AcceptanceCriterionTextUpdate
    ) -> AcceptanceCriterion | None:
        """Save a reviewer's edit; return None if the AC is missing."""
        record = self.ac_repository.get(session_id, ac_id)
        if record is None:
            return None
        updated = self.ac_repository.update_text(
            record, data.title, data.given, data.when, data.then
        )
        return self._with_alternatives(updated)

    def update_status(
        self, session_id: int, ac_id: int, data: AcceptanceCriterionStatusUpdate
    ) -> AcceptanceCriterion | None:
        """Set an AC's status; return None if the AC is missing."""
        record = self.ac_repository.get(session_id, ac_id)
        if record is None:
            return None
        updated = self.ac_repository.update_status(record, data.status)
        return self._with_alternatives(updated)

    def _with_alternatives(
        self, record: AcceptanceCriterionRecord
    ) -> AcceptanceCriterion:
        """Return one AC's wire model with its alternatives and UAT case count."""
        [criterion] = to_acceptance_criteria(
            self.ac_repository,
            record.session_id,
            [record],
            self.uat_case_repository.counts_by_ac(record.session_id),
        )
        return criterion

    def select_alternative(
        self, session_id: int, ac_id: int, candidate_id: int
    ) -> SelectAlternativeResponse | None:
        """Swap another model's version in as the displayed one.

        Return None if the AC or candidate is missing, or they don't match.
        """
        record = self.ac_repository.get(session_id, ac_id)
        candidate = self.ac_repository.get_candidate(session_id, candidate_id)
        if record is None or candidate is None:
            return None
        if candidate.criterion_position != record.position:
            return None

        if not candidate.is_winner:
            record = self.ac_repository.swap_in_candidate(record, candidate)

        return SelectAlternativeResponse(
            acceptance_criterion=self._with_alternatives(record),
            uat_cases_affected=len(self.uat_case_repository.list_for_ac(ac_id)),
        )

    def regenerate_selected(
        self, session_id: int, ac_id: int, data: RegenerateSelectedRequest
    ) -> RegenerateSelectedResponse | None:
        """Propose one refined version of an AC, scored but not saved.

        Return None if the AC is missing. Raise `GenerationError` if the model fails.
        """
        target = self.ac_repository.get(session_id, ac_id)
        if target is None:
            return None

        prompt = _build_regeneration_prompt(target, data)
        try:
            result = self.regeneration_agent.run_sync(prompt)
        except Exception as error:
            raise GenerationError(
                f"{get_models_config().chat_model.name} could not regenerate "
                f"'{target.title}': {error}"
            ) from error

        criterion = result.output
        candidate_text = render_ac_candidate(
            criterion.title, criterion.given, criterion.when, criterion.then
        )
        [score] = self.scorer(
            prompt=prompt, candidates=[candidate_text], settings=self.settings
        )

        # Placeholder ID, as nothing is stored until `apply_approved`.
        refined = AcceptanceCriterion(
            id=-1,
            title=criterion.title,
            given=criterion.given,
            when=criterion.when,
            then=criterion.then,
            scores=AcceptanceCriterionScores(
                relevance=score.relevance,
                correctness=score.correctness,
                understandability=score.understandability,
                coverage=score.coverage,
            ),
            overall_score=score.overall,
            status="pending",
        )
        reply = ConversationMessage(
            role="assistant",
            text=(
                f"Regenerated '{target.title}' using the additional context you "
                "provided."
            ),
        )
        return RegenerateSelectedResponse(reply=reply, candidates=[refined])

    def apply_approved(
        self, session_id: int, ac_id: int, data: ApplyApprovedRequest
    ) -> list[AcceptanceCriterion] | None:
        """Replace one AC with the approved candidates and delete its UAT cases.

        Return the updated AC list, or None if the session or AC is missing.
        Raise `ValueError` if no candidates were approved.
        """
        if self.session_repository.get(session_id) is None:
            return None
        if self.ac_repository.get(session_id, ac_id) is None:
            return None
        if not data.candidates:
            raise ValueError(
                "Must approve at least one candidate; use Cancel to discard instead."
            )

        # UAT cases reference this AC by foreign key, so delete them first.
        self.uat_case_repository.delete_for_ac(ac_id)
        updated = self.ac_repository.replace_one(session_id, ac_id, data.candidates)
        return to_acceptance_criteria(
            self.ac_repository,
            session_id,
            updated,
            self.uat_case_repository.counts_by_ac(session_id),
        )

    def regenerate_all_kickoff(self, session_id: int) -> MessageRead | None:
        """Post an assistant message asking for context to regenerate.

        The message names any rejected ACs. Return None if the session is missing.
        """
        chat_session = self.session_repository.get(session_id)
        if chat_session is None:
            return None

        rejected = self.ac_repository.list_rejected(session_id)
        if rejected:
            titles = ", ".join(f"'{criterion.title}'" for criterion in rejected)
            text = (
                f"I've noted you rejected: {titles}. Could you share more context — "
                "any missing edge cases, wrong assumptions, or extra information — "
                "so I can improve the acceptance criteria on the next generation?"
            )
        else:
            text = (
                "Let's regenerate the acceptance criteria from scratch. Could you "
                "share any extra context that would help — missing edge cases, "
                "assumptions, or additional information?"
            )

        message = self.message_repository.create(
            session_id=session_id, role="assistant", text=text, attachments=[]
        )
        self.session_repository.touch(chat_session)
        return MessageRead(
            id=message.id,  # type: ignore[arg-type]
            role=message.role,  # type: ignore[arg-type]
            text=message.text,
            attachments=[],
            created_at=message.created_at,
        )
