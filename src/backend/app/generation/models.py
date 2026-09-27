"""Schemas for the ensemble generation layer.

`GeneratedCriteriaSet` is the contract every generation agent must satisfy.
PydanticAI validates the LLM's response against it and retries on failure, so
malformed Gherkin never leaves an agent (backlog R7).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class GeneratedCriterion(BaseModel):
    """One Gherkin acceptance criterion produced by a generation agent.

    Field descriptions are sent to the LLM as part of the schema, so they do
    double duty as prompt guidance.
    """

    title: str = Field(
        description="Short imperative summary of the behaviour, at most 8 words",
        min_length=3,
    )
    given: str = Field(
        description="Precondition or initial system/user state. Do not include the word 'Given'.",
        min_length=3,
    )
    when: str = Field(
        description="The specific action or event being tested. Do not include the word 'When'.",
        min_length=3,
    )
    then: str = Field(
        description="The expected, observable outcome. Do not include the word 'Then'.",
        min_length=3,
    )

    def as_sentence(self) -> str:
        """Flatten to the single string the voting layer scores.

        Aaron's `EvaluationInput.output` is a `list[str]`, so the structured
        criterion has to be linearised before it crosses that boundary.
        """
        return f"Given {self.given}, when {self.when}, then {self.then}."

    def dedupe_key(self) -> str:
        """Normalised key used to drop near-identical criteria across agents.

        Each clause is lowercased, stripped of punctuation and collapsed to
        single spaces independently, then rejoined. Normalising per-clause
        rather than across the whole string stops punctuation next to a
        separator from leaving stray whitespace in the key.

        This only catches textually identical criteria — two agents phrasing
        the same behaviour differently will both survive, which is intentional
        for now (semantic dedupe is a separate piece of work).
        """
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


def title_key(title: str) -> str:
    """Join key for title-anchored grouping.

    Pass-2 agents are told to copy titles verbatim, but models still drift on
    case, punctuation and trailing full stops. Normalising the same way
    `dedupe_key` does means "Rate a finished book" and "Rate a finished book."
    land in the same group instead of forming a group of one each.
    """
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


class Candidate(BaseModel):
    """One agent's attempt at a single criterion, with its score once voted.

    Distinct from `GeneratedCriterion` because it carries the two things the
    criterion itself has no business knowing: which agent produced it, and how
    the voting layer rated it.
    """

    agent_id: str
    criterion: GeneratedCriterion
    relevance: float = 0.0
    correctness: float = 0.0
    understandability: float = 0.0
    coverage: float = 0.0
    overall_score: float = 0.0
    # The voting layer never rated this one; its 0.0 is an absence of a
    # verdict, not a bad verdict. Kept separate so ranking can say so.
    scoring_failed: bool = False


class CandidateGroup(BaseModel):
    """Every agent's take on one title, best first.

    The winner is `candidates[0]` after `rank()`; the rest are the alternatives
    the UI can offer behind it. A group with a single candidate is normal — it
    just means only one agent answered that title.
    """

    title: str
    candidates: list[Candidate] = Field(default_factory=list)

    def rank(self) -> None:
        """Sort best-first, breaking ties deterministically.

        Unrated candidates sort last regardless of their nominal 0.0. That is
        already where a 0.0 lands, but making it explicit means a group whose
        scoring failed entirely still ranks by agent_id rather than appearing
        to have been judged.

        Without the final key, equally-scored candidates resolve by whatever
        order the agents happened to finish in, so the "winner" shown to a
        reviewer could change between identical runs. Falling back to agent_id
        is arbitrary but stable, which is the property that matters.
        """
        self.candidates.sort(
            key=lambda c: (c.scoring_failed, -c.overall_score, c.agent_id)
        )

    @property
    def unrated(self) -> list[Candidate]:
        """Candidates the voting layer never actually scored."""
        return [c for c in self.candidates if c.scoring_failed]

    @property
    def winner(self) -> Candidate:
        return self.candidates[0]

    @property
    def alternatives(self) -> list[Candidate]:
        return self.candidates[1:]


class AgentResult(BaseModel):
    """One agent's outcome, including the failure case.

    A failed agent returns a result with `error` set rather than raising, so a
    missing API key or a rate limit degrades the ensemble instead of killing
    the request. This mirrors the convention already used in the voting layer.
    """

    agent_id: str
    provider: str
    model: str
    criteria: list[GeneratedCriterion] = Field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class EnsembleResult(BaseModel):
    """Everything the orchestrator produced for one generation run."""

    prompt: str
    results: list[AgentResult]

    @property
    def successful(self) -> list[AgentResult]:
        return [result for result in self.results if result.ok]

    @property
    def failed(self) -> list[AgentResult]:
        return [result for result in self.results if not result.ok]
