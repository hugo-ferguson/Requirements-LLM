"""Schemas for the ensemble generation layer.

Agents return the `*Set` models, which PydanticAI validates and retries (R7).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class GeneratedCriterion(BaseModel):
    """One Gherkin acceptance criterion produced by a generation agent.

    Field descriptions are sent to the LLM as part of the schema, so
    they do double duty as prompt guidance.
    """

    title: str = Field(
        description="Short imperative summary of the behaviour, at most 8 words",
        min_length=3,
    )
    given: str = Field(
        description=(
            "Precondition or initial system/user state. "
            "Do not include the word 'Given'."
        ),
        min_length=3,
    )
    when: str = Field(
        description=(
            "The specific action or event being tested. Do not include the word 'When'."
        ),
        min_length=3,
    )
    then: str = Field(
        description="The expected, observable outcome. Do not include the word 'Then'.",
        min_length=3,
    )

    def dedupe_key(self) -> str:
        """Return a key matching criteria that differ only in case or punctuation."""
        clauses = (self.given, self.when, self.then)
        return "|".join(
            re.sub(r"[^a-z0-9]+", " ", clause.lower()).strip() for clause in clauses
        )


class GeneratedCriteriaSet(BaseModel):
    """The full output of a single agent run."""

    user_story_summary: str = Field(
        description="One-sentence summary of the user story being addressed"
    )
    criteria: list[GeneratedCriterion] = Field(
        description="Gherkin acceptance criteria, one per distinct testable behaviour",
        min_length=1,
        max_length=10,
    )


class NumberedCriterion(BaseModel):
    """A pass-2 answer: given/when/then for one numbered anchor title.

    There is deliberately no `title` field. A title copied back can be
    reworded so that nothing joins on it (one model matched 0 of 5),
    losing all of that model's candidates. A number can't be
    paraphrased, and the anchor's exact title is attached in code.
    """

    title_number: int = Field(
        description="The number of the title this answers, as given in the list", ge=1
    )
    given: str = Field(
        description=(
            "Precondition or initial system/user state. "
            "Do not include the word 'Given'."
        ),
        min_length=3,
    )
    when: str = Field(
        description=(
            "The specific action or event being tested. Do not include the word 'When'."
        ),
        min_length=3,
    )
    then: str = Field(
        description="The expected, observable outcome. Do not include the word 'Then'.",
        min_length=3,
    )


class NumberedCriteriaSet(BaseModel):
    """The full output of a pass-2 agent run."""

    criteria: list[NumberedCriterion] = Field(
        description="One entry per numbered title, identified by its number",
        min_length=1,
        max_length=10,
    )


def title_key(title: str) -> str:
    """Return a title ignoring case and punctuation, for grouping."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


class Candidate(BaseModel):
    """One agent's attempt at one criterion, with its scores once voted."""

    agent_id: str
    criterion: GeneratedCriterion
    relevance: float = 0.0
    correctness: float = 0.0
    understandability: float = 0.0
    coverage: float = 0.0
    overall_score: float = 0.0
    # Never scored, so its 0.0 means nothing. Ranked last.
    scoring_failed: bool = False


class CandidateGroup(BaseModel):
    """Every agent's take on one title, best first after `rank()`."""

    title: str
    candidates: list[Candidate] = Field(default_factory=list)

    def rank(self) -> None:
        """Sort best-first, with unscored last and ties broken by agent ID."""
        self.candidates.sort(
            key=lambda candidate: (
                candidate.scoring_failed,
                -candidate.overall_score,
                candidate.agent_id,
            )
        )

    @property
    def unrated(self) -> list[Candidate]:
        """Candidates the voting layer never scored."""
        return [candidate for candidate in self.candidates if candidate.scoring_failed]

    @property
    def winner(self) -> Candidate:
        """The best candidate, once `rank()` has run."""
        return self.candidates[0]

    @property
    def alternatives(self) -> list[Candidate]:
        """Every candidate except the winner."""
        return self.candidates[1:]


class AgentResult(BaseModel):
    """One agent's outcome. A failed agent sets `error` instead of raising."""

    agent_id: str
    provider: str
    model: str
    criteria: list[GeneratedCriterion] = Field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the agent finished without an error."""
        return self.error is None


class EnsembleResult(BaseModel):
    """Everything the orchestrator produced for one generation run."""

    prompt: str
    results: list[AgentResult]

    @property
    def successful(self) -> list[AgentResult]:
        """The results that finished without an error."""
        return [result for result in self.results if result.ok]

    @property
    def failed(self) -> list[AgentResult]:
        """The results that ended in an error."""
        return [result for result in self.results if not result.ok]


# UAT cases use the same two passes as ACs, once per accepted AC.


class GeneratedUatCase(BaseModel):
    """One UAT test case, as written by a model."""

    title: str = Field(
        description="Short label for the scenario under test", min_length=3
    )
    description: str = Field(
        description="Concrete steps/data a tester follows and the expected result",
        min_length=3,
    )


class GeneratedUatCaseSet(BaseModel):
    """Pass-1 (anchor) output for one acceptance criterion."""

    cases: list[GeneratedUatCase] = Field(
        description="2-4 concrete UAT test cases grounded in the acceptance criterion",
        min_length=2,
        max_length=4,
    )


class NumberedUatCase(BaseModel):
    """A pass-2 answer: the description for one numbered case title."""

    title_number: int = Field(
        description=(
            "The number of the test case title this answers, as given in the list"
        ),
        ge=1,
    )
    description: str = Field(
        description="Concrete steps/data a tester follows and the expected result",
        min_length=3,
    )


class NumberedUatCaseSet(BaseModel):
    """Pass-2 output for one acceptance criterion."""

    cases: list[NumberedUatCase] = Field(
        description="One entry per numbered test case title, identified by its number",
        min_length=1,
        max_length=10,
    )


class UatAgentResult(BaseModel):
    """One agent's UAT outcome for one AC, including failure."""

    agent_id: str
    provider: str
    model: str
    cases: list[GeneratedUatCase] = Field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the agent finished without an error."""
        return self.error is None


class UatCandidate(BaseModel):
    """One agent's version of one test case, with its scores once voted."""

    agent_id: str
    case: GeneratedUatCase
    relevance: float = 0.0
    correctness: float = 0.0
    understandability: float = 0.0
    coverage: float = 0.0
    overall_score: float = 0.0
    scoring_failed: bool = False


class UatCandidateGroup(BaseModel):
    """Every agent's version of one test case title, best first."""

    title: str
    candidates: list[UatCandidate] = Field(default_factory=list)

    def rank(self) -> None:
        """Sort best-first, the same way as `CandidateGroup.rank`."""
        self.candidates.sort(
            key=lambda candidate: (
                candidate.scoring_failed,
                -candidate.overall_score,
                candidate.agent_id,
            )
        )

    @property
    def winner(self) -> UatCandidate:
        """The best candidate, once `rank()` has run."""
        return self.candidates[0]

    @property
    def alternatives(self) -> list[UatCandidate]:
        """Every candidate except the winner."""
        return self.candidates[1:]
