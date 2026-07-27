from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationInput(BaseModel):
    ai: str
    model: str
    prompt: str
    output: list[str] = Field(min_length=1)
    reference_answer: str = ""


class PrometheusVote(BaseModel):
    feedback: str
    score: int = Field(ge=1, le=5)


class Vote(PrometheusVote):
    output_index: int
    output: str
    rubric: str


class VotingResult(BaseModel):
    ai: str
    model: str
    votes: list[Vote]
