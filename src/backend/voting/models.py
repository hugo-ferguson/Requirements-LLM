from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# The app's model config. The voting package reads its `judges` section only
# when run standalone (the scripts here, the MCP server); the app passes
# judges in explicitly.
DEFAULT_MODELS_CONFIG = Path(__file__).resolve().parent.parent / "config" / "models.json"


class JudgeConfig(BaseModel):
    """One scoring judge, as declared under `judges` in config/models.json.

    The fields after `model` say how the judge's API differs from LiteLLM's
    defaults. They are set per judge rather than inferred from the model
    name, because LiteLLM's own capability data gets them wrong: it reports
    Sonnet 5.5 as accepting `tool_choice` and `temperature`, and the API
    rejects both.
    """

    id: str = Field(description="Label used in logs and results, e.g. 'claude'")
    model: str = Field(description="LiteLLM model string, e.g. 'anthropic/claude-sonnet-5-5'")
    enabled: bool = True
    # One call scoring all four rubrics, or Prometheus-style, one call each.
    style: Literal["combined", "per_rubric"] = "combined"
    # None sends no temperature at all. Claude from Opus 4.7 / Sonnet 5 on and
    # OpenAI reasoning models reject any non-default value.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Send `response_format`. LiteLLM implements it for Anthropic as a forced
    # tool call, which Claude from Sonnet 5.5 / Opus 5.5 on rejects; without
    # it the schema in the system prompt is enough.
    structured_output: bool = True
    # Mark the shared rubric prefix with `cache_control`. Anthropic needs the
    # marker; OpenAI and Gemini cache long prefixes on their own.
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
