"""Data models for judges, voting inputs, votes and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from llm.spec import ModelSpec

# Judges are read from here only when run standalone. The app passes them in.
DEFAULT_MODELS_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "models.json"
)


class JudgeConfig(ModelSpec):
    """One scoring judge, from `judges` in config/models.json."""

    id: str = Field(description="Label used in logs and results, e.g. 'claude'")
    enabled: bool = True
    # One call for all four rubrics, or one call per rubric.
    style: Literal["combined", "per_rubric"] = "combined"
    # Cache the shared prompt prefix, scoring one candidate first to fill it.
    # Only Anthropic needs this; OpenAI and Gemini cache on their own.
    cache_prompt: bool = False
    # Concurrent requests. Use 1 for Ollama, which serves one at a time.
    max_parallel: int = Field(default=8, ge=1)

    @property
    def display_name(self) -> str:
        """The id made readable: "local-qwen" -> "Local Qwen"."""
        return self.id.replace("-", " ").replace("_", " ").title()


def load_judges(path: Path = DEFAULT_MODELS_CONFIG) -> list[JudgeConfig]:
    """Return the enabled judges from a models config file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    judges = [JudgeConfig.model_validate(entry) for entry in payload.get("judges", [])]
    return [judge for judge in judges if judge.enabled]


class EvaluationInput(BaseModel):
    """One prompt's candidate outputs, and the judges to score them."""

    ai: str
    model: str
    prompt: str
    output: list[str] = Field(min_length=1)
    reference_answer: str = ""
    judges: list[JudgeConfig] = Field(default_factory=load_judges, min_length=1)


class PrometheusVote(BaseModel):
    """A judge's score and feedback for one rubric."""

    feedback: str
    # -1 indicates an evaluation error; normal scores range from 1 to 5.
    score: int = Field(ge=-1, le=5)


class CombinedVote(BaseModel):
    """A combined-style judge's votes on all four rubrics at once."""

    correctness: PrometheusVote
    coverage: PrometheusVote
    relevance: PrometheusVote
    understandability: PrometheusVote


class RubricFeedback(BaseModel):
    """One judge's value and feedback for one rubric."""

    rubric: str
    value: int
    feedback: str


class ProviderFeedback(BaseModel):
    """Everything one judge said about one output."""

    ai: str
    model: str
    feedback: list[RubricFeedback]
    overall_score: float


class RubricAverage(BaseModel):
    """One rubric averaged across judges; -1 if none rated it."""

    rubric: str
    value: float


class EvaluatedOutput(BaseModel):
    """One candidate output with every judge's feedback and the averages."""

    output: str
    feedback: list[ProviderFeedback]
    rubric_averages: list[RubricAverage]
    overall_score: float


class VotingResult(BaseModel):
    """The judges' verdict on every candidate output for one input."""

    ai: str
    model: str
    prompt: str
    output: list[EvaluatedOutput]
    overall_score: float = -1.0
    rank: int = 0
