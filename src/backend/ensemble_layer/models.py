"""
Pydantic data models for the Ensemble Generation Layer.

These models define the contracts between layers:
  - Input  : UserStory  -> what each Generation Agent receives
  - Output : AgentResult -> what each Generation Agent produces
  - Bundle : EnsembleResult -> the full set of N agent outputs
    passed downstream to the Voting Layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# -------------
# Enums
# -------------

class GenerationStatus(str, Enum):
    SUCCESS = "success"
    FAILED  = "failed"
    TIMEOUT = "timeout"


class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


# ----------------
# Input models
# ----------------

class UserStory(BaseModel):
    """
    The primary input unit fed into the Ensemble Generation Layer.
    Mirrors what arrives from the Input Layer (text + optional UI context).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Short title for the user story")
    description: str = Field(..., description="Full user story text, e.g. 'As a user, I want …'")
    additional_context: str | None = Field(
        None,
        description="Extra BE/domain context injected by the orchestrator (black-box logic, API contracts, etc.)"
    )
    ui_context: str | None = Field(
        None,
        description="Extracted text from UI mockups via Vision-to-Text (from Input Layer)"
    )


class AgentConfig(BaseModel):
    """Configuration for a single Generation Agent."""
    agent_id: int
    provider: Provider
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024)


class EnsembleConfig(BaseModel):
    """Runtime configuration for one ensemble run."""
    agents: list[AgentConfig]
    output_format: str = Field(default="gherkin", description="'gherkin' or 'plain'")

    @property
    def num_agents(self) -> int:
        return len(self.agents)


# ----------------
# Output models
# ----------------

class AcceptanceCriteria(BaseModel):
    """A single acceptance criterion produced by a generation agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_title: str
    given: list[str] = Field(default_factory=list)
    when: list[str] = Field(default_factory=list)
    then: list[str] = Field(default_factory=list)
    raw_text: str = Field("", description="Verbatim text from the LLM before parsing")

    def to_gherkin(self) -> str:
        lines = [f"  Scenario: {self.scenario_title}"]
        for g in self.given:
            lines.append(f"    Given {g}")
        for w in self.when:
            lines.append(f"    When {w}")
        for t in self.then:
            lines.append(f"    Then {t}")
        return "\n".join(lines)


class AgentResult(BaseModel):
    """Output from a single Generation Agent."""
    agent_id: int
    user_story_id: str
    status: GenerationStatus = GenerationStatus.SUCCESS
    criteria: list[AcceptanceCriteria] = Field(default_factory=list)
    raw_response: str = ""
    error_message: str | None = None
    temperature_used: float = 0.7
    latency_ms: float | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_gherkin_feature(self) -> str:
        if self.status != GenerationStatus.SUCCESS:
            return f"# Agent {self.agent_id} failed: {self.error_message}"
        feature_lines = [
            f"Feature: Acceptance Criteria (Agent #{self.agent_id})",
            "",
        ]
        for ac in self.criteria:
            feature_lines.append(ac.to_gherkin())
            feature_lines.append("")
        return "\n".join(feature_lines)


class EnsembleResult(BaseModel):
    """
    The complete output of one Ensemble Generation Layer run.
    This is the payload passed to the Voting Layer.
    """
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_story: UserStory
    config: EnsembleConfig
    agent_results: list[AgentResult] = Field(default_factory=list)
    total_latency_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def successful_results(self) -> list[AgentResult]:
        return [r for r in self.agent_results if r.status == GenerationStatus.SUCCESS]

    @property
    def success_rate(self) -> float:
        if not self.agent_results:
            return 0.0
        return len(self.successful_results) / len(self.agent_results)

    def summary(self) -> str:
        return (
            f"Run {self.run_id[:8]} | "
            f"Agents: {len(self.agent_results)} | "
            f"Success: {len(self.successful_results)} | "
            f"Latency: {self.total_latency_ms:.0f}ms"
        )