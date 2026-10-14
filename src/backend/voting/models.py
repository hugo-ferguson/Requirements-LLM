"""Data models for judges, voting inputs, votes and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from llm.spec import ModelSpec


# The app's model config. The voting package reads its `judges` section
# only when run standalone (the scripts here, the MCP server); the app
# passes judges in explicitly.
DEFAULT_MODELS_CONFIG = Path(__file__).resolve(
).parent.parent / "config" / "models.json"


class JudgeConfig(ModelSpec):
    """One scoring judge, from `judges` in config/models.json.

    The model fields (`provider`, `model`, `temperature`, `output_mode`,
    ...) are the same as every other entry in that file; the rest say
    how the judge votes.
    """

    id: str = Field(
        description="Label used in logs and results, e.g. 'claude'")
    enabled: bool = True
    # One call scoring all four rubrics, or Prometheus-style, one call
    # each.
    style: Literal["combined", "per_rubric"] = "combined"
    # Mark the shared rubric prefix for caching, and score one candidate
    # before the rest so they read the cache it writes. Anthropic needs
    # the marker; OpenAI and Gemini cache long prefixes on their own.
    cache_prompt: bool = False
    # Requests in flight at once. 1 for Ollama, which serves one request
    # per model at a time, so extra concurrency only queues against the
    # timeout.
    max_parallel: int = Field(default=8, ge=1)

    @property
    def display_name(self) -> str:
        """The id made readable: "local-qwen" -> "Local Qwen"."""
        return self.id.replace("-", " ").replace("_", " ").title()


def load_judges(path: Path = DEFAULT_MODELS_CONFIG) -> list[JudgeConfig]:
    """Return the enabled judges from a models config file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    judges = [JudgeConfig.model_validate(item)
              for item in payload.get("judges", [])]
    return [judge for judge in judges if judge.enabled]


class EvaluationInput(BaseModel):
    """One prompt's candidate outputs, and the judges to score them."""

    ai: str
    model: str
    prompt: str
    output: list[str] = Field(min_length=1)
    reference_answer: str = ""
    judges: list[JudgeConfig] = Field(
        default_factory=load_judges, min_length=1)


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


class Vote(PrometheusVote):
    """A rubric vote tied to the output it scored."""

    output_index: int
    output: str
    rubric: str


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
    """One candidate output with every judge's feedback and averages."""

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
