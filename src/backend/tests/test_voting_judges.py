"""Several judges score every candidate and their scores are averaged."""

from __future__ import annotations

import asyncio

import app.services.scoring as scoring
import voting.voting as voting
from app.config import Settings
from voting.models import (
    EvaluatedOutput,
    EvaluationInput,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)

RUBRICS = ("correctness", "coverage", "relevance", "understandability")


def _judged(evaluation_input: EvaluationInput, judge: str, score: int) -> VotingResult:
    """What one judge returns: every rubric of every candidate at `score`."""
    return VotingResult(
        ai=evaluation_input.ai,
        model=evaluation_input.model,
        prompt=evaluation_input.prompt,
        output=[
            EvaluatedOutput(
                output=output,
                feedback=[
                    ProviderFeedback(
                        ai=judge,
                        model=judge,
                        feedback=[
                            RubricFeedback(rubric=r, value=score, feedback="") for r in RUBRICS
                        ],
                        overall_score=float(score),
                    )
                ],
                rubric_averages=[RubricAverage(rubric=r, value=float(score)) for r in RUBRICS],
                overall_score=float(score),
            )
            for output in evaluation_input.output
        ],
    )


def test_judge_names_come_from_settings() -> None:
    assert Settings(voting_judges="claude, Luna ,").voting_judge_names == ["claude", "luna"]
    assert Settings(voting_judges="").voting_judge_names == ["claude"]


def test_score_candidates_asks_every_configured_judge(monkeypatch) -> None:
    seen: list[list[str]] = []

    async def fake_evaluate_input(evaluation_input):
        seen.append(evaluation_input.providers)
        return _judged(evaluation_input, "any", 4)

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    scoring.score_candidates("prompt", ["a"], settings=Settings(voting_judges="claude,luna"))

    assert seen == [["claude", "luna"]]


def test_two_judges_are_averaged(monkeypatch) -> None:
    scores = {"claude": 4, "luna": 2}

    async def judge(name, evaluation_input):
        return _judged(evaluation_input, name, scores[name])

    monkeypatch.setattr(voting, "_evaluate_with_provider", judge)
    evaluation_input = EvaluationInput(
        ai="t", model="t", prompt="p", output=["a"], providers=["claude", "luna"]
    )

    [result] = asyncio.run(voting.evaluate_input(evaluation_input)).output

    assert result.overall_score == 3.0
    assert {avg.rubric: avg.value for avg in result.rubric_averages}["coverage"] == 3.0


def test_a_failing_judge_is_averaged_out_and_the_candidate_is_still_rated(monkeypatch) -> None:
    async def judge(name, evaluation_input):
        if name == "luna":
            raise RuntimeError("luna is down")
        return _judged(evaluation_input, name, 4)

    monkeypatch.setattr(voting, "_evaluate_with_provider", judge)
    monkeypatch.setattr(scoring, "_evaluate_input", voting.evaluate_input)

    [score] = scoring.score_candidates(
        "prompt", ["a"], settings=Settings(voting_judges="claude,luna")
    )

    assert score.failed is False
    assert score.overall == 4.0
