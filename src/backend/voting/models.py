from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from llm.spec import ModelSpec


# The app's model config. The voting package reads its `judges` section only
# when run standalone (the scripts here, the MCP server); the app passes
# judges in explicitly.
DEFAULT_MODELS_CONFIG = Path(__file__).resolve().parent.parent / "config" / "models.json"


class JudgeConfig(ModelSpec):
    """One scoring judge, as declared under `judges` in config/models.json.

    The model fields (`provider`, `model`, `temperature`, `output_mode`, ...)
    are the same as every other entry in that file; the rest say how the
    judge votes.
    """

    id: str = Field(description="Label used in logs and results, e.g. 'claude'")
    enabled: bool = True
    # One call scoring all four rubrics, or Prometheus-style, one call each.
    style: Literal["combined", "per_rubric"] = "combined"
    # Mark the shared rubric prefix for caching, and score one candidate before
    # the rest so they read the cache it writes. Anthropic needs the marker;
    # OpenAI and Gemini cache long prefixes on their own.
    cache_prompt: bool = False
    # Requests in flight at once. 1 for Ollama, which serves one request per
    # model at a time, so extra concurrency only queues against the timeout.
    max_parallel: int = Field(default=8, ge=1)

    @property
    def display_name(self) -> str:
        return self.id.replace("-", " ").replace("_", " ").title()


def load_judges(path: Path = DEFAULT_MODELS_CONFIG) -> list[JudgeConfig]:
    """The enabled judges from a models config file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    judges = [JudgeConfig.model_validate(item) for item in payload.get("judges", [])]
    return [judge for judge in judges if judge.enabled]


class EvaluationInput(BaseModel):
    ai: str
    model: str
    prompt: str
    output: list[str] = Field(min_length=1)
    reference_answer: str = ""
    judges: list[JudgeConfig] = Field(default_factory=load_judges, min_length=1)


class PrometheusVote(BaseModel):
    feedback: str
    # -1 indicates an evaluation error; normal scores range from 1 to 5.
    score: int = Field(ge=-1, le=5)


class CombinedVote(BaseModel):
    correctness: PrometheusVote
    coverage: PrometheusVote
    relevance: PrometheusVote
    understandability: PrometheusVote


class Vote(PrometheusVote):
    output_index: int
    output: str
    rubric: str


class RubricFeedback(BaseModel):
    rubric: str
    value: int
    feedback: str


class ProviderFeedback(BaseModel):
    ai: str
    model: str
    feedback: list[RubricFeedback]
    overall_score: float


class RubricAverage(BaseModel):
    rubric: str
    value: float


class EvaluatedOutput(BaseModel):
    output: str
    feedback: list[ProviderFeedback]
    rubric_averages: list[RubricAverage]
    overall_score: float


class VotingResult(BaseModel):
    ai: str
    model: str
    prompt: str
    output: list[EvaluatedOutput]
    overall_score: float = -1.0
    rank: int = 0
