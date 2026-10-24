"""Several judges score every candidate; their scores are averaged."""

from __future__ import annotations

import asyncio

import app.services.scoring as scoring
import voting.voting as voting
from voting.models import (
    EvaluatedOutput,
    EvaluationInput,
    JudgeConfig,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)

RUBRICS = ("correctness", "coverage", "relevance", "understandability")
CLAUDE = JudgeConfig(id="claude", provider="anthropic", model="claude-sonnet-5-5")
LUNA = JudgeConfig(id="luna", provider="openai", model="gpt-6-luna")


def _judged(
    evaluation_input: EvaluationInput, judge_name: str, score: int
) -> VotingResult:
    """Return one judge's result, with every rubric at `score`."""
    return VotingResult(
        ai=evaluation_input.ai,
        model=evaluation_input.model,
        prompt=evaluation_input.prompt,
        output=[
            EvaluatedOutput(
                output=output,
                feedback=[
                    ProviderFeedback(
                        ai=judge_name,
                        model=judge_name,
                        feedback=[
                            RubricFeedback(rubric=rubric, value=score, feedback="")
                            for rubric in RUBRICS
                        ],
                        overall_score=float(score),
                    )
                ],
                rubric_averages=[
                    RubricAverage(rubric=rubric, value=float(score))
                    for rubric in RUBRICS
                ],
                overall_score=float(score),
            )
            for output in evaluation_input.output
        ],
    )


def test_score_candidates_uses_the_enabled_judges_from_the_models_config(
    monkeypatch,
) -> None:
    seen: list[list[str]] = []

    async def fake_evaluate_input(evaluation_input):
        seen.append([judge.id for judge in evaluation_input.judges])
        return _judged(evaluation_input, "any", 4)

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    # tests/models.test.json declares one judge, `claude`.
    scoring.score_candidates("prompt", ["a"])

    assert seen == [["claude"]]


def test_score_candidates_asks_every_given_judge(monkeypatch) -> None:
    seen: list[list[str]] = []

    async def fake_evaluate_input(evaluation_input):
        seen.append([judge.id for judge in evaluation_input.judges])
        return _judged(evaluation_input, "any", 4)

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    scoring.score_candidates("prompt", ["a"], judges=[CLAUDE, LUNA])

    assert seen == [["claude", "luna"]]


def test_two_judges_are_averaged(monkeypatch) -> None:
    scores = {"claude": 4, "luna": 2}

    async def fake_evaluate_with_judge(judge, evaluation_input):
        return _judged(evaluation_input, judge.id, scores[judge.id])

    monkeypatch.setattr(voting, "_evaluate_with_judge", fake_evaluate_with_judge)
    evaluation_input = EvaluationInput(
        ai="t", model="t", prompt="p", output=["a"], judges=[CLAUDE, LUNA]
    )

    [result] = asyncio.run(voting.evaluate_input(evaluation_input)).output

    assert result.overall_score == 3.0
    averages = {average.rubric: average.value for average in result.rubric_averages}
    assert averages["coverage"] == 3.0


def test_a_failing_judge_is_averaged_out_and_the_candidate_is_still_rated(
    monkeypatch,
) -> None:
    async def fake_evaluate_with_judge(judge, evaluation_input):
        if judge.id == "luna":
            raise RuntimeError("luna is down")
        return _judged(evaluation_input, judge.id, 4)

    monkeypatch.setattr(voting, "_evaluate_with_judge", fake_evaluate_with_judge)
    monkeypatch.setattr(scoring, "_evaluate_input", voting.evaluate_input)

    [score] = scoring.score_candidates("prompt", ["a"], judges=[CLAUDE, LUNA])

    assert score.failed is False
    assert score.overall == 4.0


def test_every_scoring_call_runs_on_one_long_lived_loop(monkeypatch) -> None:
    """Share one loop so judge concurrency limits aren't multiplied."""
    import threading

    loops = []

    async def fake_evaluate_input(evaluation_input):
        loops.append((asyncio.get_running_loop(), threading.current_thread()))
        return _judged(evaluation_input, "any", 4)

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    scoring.score_candidates("prompt", ["a"], judges=[CLAUDE])
    scoring.score_candidates("prompt", ["b"], judges=[CLAUDE])

    (first_loop, first_thread), (second_loop, _) = loops
    assert first_loop is second_loop
    assert not first_loop.is_closed()
    assert first_thread is not threading.main_thread()
