"""Business logic for generating and reviewing UAT test cases."""

import asyncio
from collections.abc import Awaitable, Callable

from pydantic_ai import Agent

from app.config import Settings
from app.config import settings as default_settings
from app.generation.models import (
    GeneratedUatCase,
    UatAgentResult,
    UatCandidate,
    UatCandidateGroup,
)
from app.generation.orchestrator import run_uat_ensemble
from app.llm_config import get_models_config
from app.models_acceptance_criteria import AcceptanceCriterionRecord
from app.models_conversation import ConversationMessage, ConversationRequest
from app.models_uat_cases import (
    UatApplyApprovedRequest,
    UatCase,
    UatCaseAlternative,
    UatCaseCandidateRecord,
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
from app.services.acceptance_criteria import to_acceptance_criterion
from app.services.agents import GenerationError, build_agent
from app.services.conversation import render_conversation
from app.services.scoring import (
    Scorer,
    render_ac_candidate,
    render_uat_candidate,
    score_candidates,
)

# Signature of `run_uat_ensemble`, so tests can pass a fake.
UatEnsemble = Callable[
    ..., Awaitable[tuple[list[UatAgentResult], list[UatCandidateGroup]]]
]


UAT_REGENERATION_SYSTEM_PROMPT = """\
You are a QA engineer refining ONE existing UAT test case using a reviewer's
follow-up feedback.

You will be given the parent acceptance criterion, the test case as it
stands now, and a conversation of additional context or feedback from the
reviewer. Produce exactly one improved test case covering the same
underlying scenario — do not invent an unrelated new one. If the reviewer's
feedback conflicts with what's given, follow the feedback; it is the more
recent and more specific instruction.
"""


def _to_uat_case(
    record: UatCaseRecord, candidates: list[UatCaseCandidateRecord] | None = None
) -> UatCase:
    """Build one case's wire model, with its alternatives attached."""
    candidates = candidates or []
    winner = next((candidate for candidate in candidates if candidate.is_winner), None)
    return UatCase(
        id=record.id,  # type: ignore[arg-type]
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
        status=record.status,  # type: ignore[arg-type]
        source_agent=winner.source_agent if winner else None,
        alternatives=[
            UatCaseAlternative(
                candidate_id=candidate.id,
                description=candidate.description,
                scores=UatCaseScores(
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
    )


def _scores_of(candidate: UatCandidate) -> UatCaseScores:
    """Return a ranked candidate's rubric scores in wire form."""
    return UatCaseScores(
        relevance=candidate.relevance,
        correctness=candidate.correctness,
        understandability=candidate.understandability,
        coverage=candidate.coverage,
    )


def _build_regeneration_prompt(
    ac: AcceptanceCriterionRecord,
    target: UatCaseRecord,
    data: UatRegenerateSelectedRequest,
) -> str:
    """Build the prompt from the parent AC, the case and any reviewer feedback."""
    parts = [
        "Parent acceptance criterion:",
        render_ac_candidate(ac.title, ac.given, ac.when, ac.then),
        "",
        "Current UAT test case:",
        render_uat_candidate(target.title, target.description),
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


class UatCaseService:
    """Generate, review and regenerate a session's UAT test cases.

    Cases are written from accepted ACs, by the same agents as AC generation.
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        uat_case_repository: UatCaseRepository,
        ac_repository: AcceptanceCriteriaRepository,
        settings: Settings | None = None,
        regeneration_agent: Agent[None, GeneratedUatCase] | None = None,
        scorer: Scorer = score_candidates,
        uat_ensemble: UatEnsemble = run_uat_ensemble,
    ):
        """Store the dependencies. Tests can swap the agent, scorer and ensemble."""
        self.session_repository = session_repository
        self.uat_case_repository = uat_case_repository
        self.ac_repository = ac_repository
        self.settings = settings or default_settings
        # Built on first use, so a missing API key only breaks regeneration.
        self._regeneration_agent = regeneration_agent
        self.scorer = scorer
        self.uat_ensemble = uat_ensemble

    @property
    def regeneration_agent(self) -> Agent[None, GeneratedUatCase]:
        """The regeneration agent, built on first use."""
        if self._regeneration_agent is None:
            self._regeneration_agent = build_agent(
                GeneratedUatCase, UAT_REGENERATION_SYSTEM_PROMPT
            )
        return self._regeneration_agent

    async def _generate_cases(self, ac: AcceptanceCriterionRecord) -> list[UatCase]:
        """Run the ensemble for one AC and return its scored cases.

        Raise `GenerationError` if no agent returned any cases.
        """
        prompt = render_ac_candidate(ac.title, ac.given, ac.when, ac.then)
        results, groups = await self.uat_ensemble(
            prompt, timeout_seconds=self.settings.generation_timeout_seconds
        )
        if not groups:
            reasons = "; ".join(
                f"{result.agent_id}: {result.error or 'no cases'}" for result in results
            )
            raise GenerationError(
                "Every generation agent failed to write UAT cases for "
                f"'{ac.title}'. {reasons}"
            )

        candidates = [candidate for group in groups for candidate in group.candidates]
        candidate_texts = [
            render_uat_candidate(candidate.case.title, candidate.case.description)
            for candidate in candidates
        ]
        scores = await asyncio.to_thread(
            self.scorer,
            prompt=prompt,
            candidates=candidate_texts,
            settings=self.settings,
        )
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.relevance = score.relevance
            candidate.correctness = score.correctness
            candidate.understandability = score.understandability
            candidate.coverage = score.coverage
            candidate.overall_score = score.overall
            candidate.scoring_failed = score.failed

        cases: list[UatCase] = []
        for index, group in enumerate(groups):
            group.rank()
            winner = group.winner
            cases.append(
                UatCase(
                    # Placeholder ID until the database assigns one.
                    id=-(index + 1),
                    ac_id=ac.id,  # type: ignore[arg-type]
                    title=group.title,
                    description=winner.case.description,
                    scores=_scores_of(winner),
                    overall_score=winner.overall_score,
                    status="pending",
                    source_agent=winner.agent_id,
                    alternatives=[
                        UatCaseAlternative(
                            description=alternative.case.description,
                            scores=_scores_of(alternative),
                            overall_score=alternative.overall_score,
                            source_agent=alternative.agent_id,
                        )
                        for alternative in group.alternatives
                    ],
                )
            )
        return cases

    def _build_group(
        self,
        ac_record: AcceptanceCriterionRecord,
        candidates: dict[tuple[int, int], list[UatCaseCandidateRecord]] | None = None,
    ) -> UatCaseGroup:
        """Return one AC and its cases.

        The AC is just a header, without alternatives or UAT case count.
        Pass `candidates` to reuse one query across groups.
        """
        if candidates is None:
            candidates = self.uat_case_repository.candidates_by_case(
                ac_record.session_id
            )
        records = self.uat_case_repository.list_for_ac(ac_record.id)  # type: ignore[arg-type]
        return UatCaseGroup(
            ac=to_acceptance_criterion(ac_record),
            uat_cases=[
                _to_uat_case(record, candidates.get((record.ac_id, record.position)))
                for record in records
            ],
        )

    def _with_alternatives(self, record: UatCaseRecord) -> UatCase:
        """Return one case's wire model with its alternatives."""
        candidates = self.uat_case_repository.candidates_by_case(record.session_id)
        return _to_uat_case(record, candidates.get((record.ac_id, record.position)))

    def list_for_session(self, session_id: int) -> UatCaseGroupsResult | None:
        """Return a session's cases grouped by AC, or None if the session is missing.

        Only ACs with cases get a group, in display order.
        """
        if self.session_repository.get(session_id) is None:
            return None

        all_cases = self.uat_case_repository.list_for_session(session_id)
        ac_ids = {case.ac_id for case in all_cases}
        ac_records = [self.ac_repository.get(session_id, ac_id) for ac_id in ac_ids]
        present_records = [record for record in ac_records if record is not None]
        present_records.sort(key=lambda record: record.position)
        candidates = self.uat_case_repository.candidates_by_case(session_id)
        groups = [self._build_group(record, candidates) for record in present_records]
        return UatCaseGroupsResult(groups=groups)

    async def generate(self, session_id: int) -> UatCaseGroupsResult | None:
        """Generate cases for every accepted AC, replacing old ones.

        Return None if the session is missing. Raise `GenerationError` if any AC
        gets no cases, before anything is stored.
        """
        if self.session_repository.get(session_id) is None:
            return None

        accepted = [
            record
            for record in self.ac_repository.list_for_session(session_id)
            if record.status == "accepted"
        ]
        # Run every AC's ensemble at once. Any failure stops before storing.
        cases_per_ac = await asyncio.gather(
            *(self._generate_cases(ac) for ac in accepted)
        )
        self.uat_case_repository.persist_batch(
            session_id,
            {
                ac.id: cases
                for ac, cases in zip(accepted, cases_per_ac, strict=True)  # type: ignore[arg-type]
            },
        )
        return self.list_for_session(session_id)

    def select_alternative(
        self, session_id: int, uat_id: int, candidate_id: int
    ) -> UatCase | None:
        """Swap another model's version in as the displayed one.

        Return None if the case or candidate is missing, or they don't match.
        """
        record = self.uat_case_repository.get(session_id, uat_id)
        candidate = self.uat_case_repository.get_candidate(session_id, candidate_id)
        if record is None or candidate is None:
            return None
        if (candidate.ac_id, candidate.case_position) != (
            record.ac_id,
            record.position,
        ):
            return None
        if not candidate.is_winner:
            record = self.uat_case_repository.swap_in_candidate(record, candidate)
        return self._with_alternatives(record)

    def update_text(
        self, session_id: int, uat_id: int, data: UatCaseTextUpdate
    ) -> UatCase | None:
        """Save a reviewer's edit; return None if the case is missing."""
        record = self.uat_case_repository.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat_case_repository.update_text(
            record, data.title, data.description
        )
        return self._with_alternatives(updated)

    def update_status(
        self, session_id: int, uat_id: int, data: UatCaseStatusUpdate
    ) -> UatCase | None:
        """Set a case's status; return None if the case is missing."""
        record = self.uat_case_repository.get(session_id, uat_id)
        if record is None:
            return None
        updated = self.uat_case_repository.update_status(record, data.status)
        return self._with_alternatives(updated)

    def regenerate_selected(
        self, session_id: int, uat_id: int, data: UatRegenerateSelectedRequest
    ) -> UatRegenerateSelectedResponse | None:
        """Propose one refined version of a case, scored but not saved.

        Return None if the case is missing. Raise `GenerationError` if its AC is
        missing or the model fails.
        """
        target = self.uat_case_repository.get(session_id, uat_id)
        if target is None:
            return None
        ac = self.ac_repository.get(session_id, target.ac_id)
        if ac is None:
            raise GenerationError(
                f"UAT case '{target.title}' has no parent acceptance criterion."
            )

        prompt = _build_regeneration_prompt(ac, target, data)
        try:
            result = self.regeneration_agent.run_sync(prompt)
        except Exception as error:
            raise GenerationError(
                f"{get_models_config().chat_model.name} could not regenerate "
                f"'{target.title}': {error}"
            ) from error

        case = result.output
        candidate_text = render_uat_candidate(case.title, case.description)
        [score] = self.scorer(
            prompt=prompt, candidates=[candidate_text], settings=self.settings
        )

        # Placeholder ID, as nothing is stored until `apply_approved`.
        refined = UatCase(
            id=-1,
            ac_id=target.ac_id,
            title=case.title,
            description=case.description,
            scores=UatCaseScores(
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
        return UatRegenerateSelectedResponse(reply=reply, candidates=[refined])

    def apply_approved(
        self, session_id: int, uat_id: int, data: UatApplyApprovedRequest
    ) -> UatCaseGroup | None:
        """Replace one case with the approved candidates, in its place.

        Return the parent AC's updated group, or None if the session, case or AC
        is missing. Raise `ValueError` if no candidates were approved.
        """
        if self.session_repository.get(session_id) is None:
            return None
        target = self.uat_case_repository.get(session_id, uat_id)
        if target is None:
            return None
        if not data.candidates:
            raise ValueError(
                "Must approve at least one candidate; use Cancel to discard instead."
            )

        # Read now, as the row is deleted by replace_one (ObjectDeletedError).
        ac_id = target.ac_id
        self.uat_case_repository.replace_one(session_id, ac_id, uat_id, data.candidates)
        ac_record = self.ac_repository.get(session_id, ac_id)
        if ac_record is None:
            return None
        return self._build_group(ac_record)
