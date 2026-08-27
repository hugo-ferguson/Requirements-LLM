from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProviderName = Literal["prometheus", "qwen", "llama"]


class EvaluationInput(BaseModel):
    ai: str
    model: str
    prompt: str
    output: list[str] = Field(min_length=1)
    reference_answer: str = ""
    providers: list[ProviderName] = Field(default_factory=lambda: ["prometheus", "qwen", "llama"], min_length=1)


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
