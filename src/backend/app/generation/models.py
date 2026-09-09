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
