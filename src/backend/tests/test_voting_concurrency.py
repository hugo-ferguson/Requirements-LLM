"""Concurrency bounds on the voting layer.

Regression cover for a bug seen in a live run: 3 of 5 candidates timed out at
exactly 120.0s while the first two scored fine. Ollama serves one request per
model at a time, so an unbounded `asyncio.gather` queues the rest — and the
per-request timeout counts that queue time, so later candidates expire before
they ever start. Downstream that reads as an unrated candidate, which always
loses its group, so a capacity problem silently picked the winner.
"""

from __future__ import annotations

import asyncio

import pytest

from voting.models import EvaluationInput, JudgeConfig
from voting.provider import (
    default_timeout,
    evaluate_with_combined_model,
    max_parallel_requests,
)


OLLAMA = JudgeConfig(id="qwen", model="ollama/qwen2.5:7b", max_parallel=1)
CLOUD = JudgeConfig(id="luna", model="openai/gpt-6-luna")


class _RecordingClient:
    """Stands in for LiteLLMCombinedClient, tracking overlap."""

    def __init__(self, judge: JudgeConfig, delay: float = 0.02) -> None:
        self.judge = judge
        self.provider_name = "Test"
        self.model = judge.model
        self.delay = delay
        self.in_flight = 0
        self.peak_in_flight = 0

    async def evaluate(self, *, instruction: str, response: str):
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1

        class _Rubric:
            score = 4
            feedback = "fine"

        class _Vote:
            correctness = coverage = relevance = understandability = _Rubric()

        return _Vote()


def _input(candidates: list[str]) -> EvaluationInput:
    return EvaluationInput(
        ai="requirements-llm",
        model="test",
        prompt="a user story",
        output=candidates,
        reference_answer="",
        judges=[JudgeConfig(id="test", model="test/test")],
    )


def test_a_judge_limited_to_one_request_is_scored_one_at_a_time():
    client = _RecordingClient(OLLAMA)

    result = asyncio.run(
        evaluate_with_combined_model(_input(["a", "b", "c", "d", "e"]), client)
    )

    assert len(result.output) == 5
    assert client.peak_in_flight == 1, (
        "concurrent requests to a single-model Ollama queue up, and the "
        "per-request timeout counts the wait"
    )


def test_cloud_judges_still_run_concurrently():
    """The bound is set per judge for local single-model servers, not as a blanket cap."""
    client = _RecordingClient(CLOUD)

    asyncio.run(evaluate_with_combined_model(_input(["a", "b", "c", "d"]), client))

    assert client.peak_in_flight > 1


def test_every_candidate_is_still_scored_when_serialised():
    client = _RecordingClient(OLLAMA)

    result = asyncio.run(evaluate_with_combined_model(_input(["a", "b", "c"]), client))

    assert [item.output for item in result.output] == ["a", "b", "c"]
    assert all(item.overall_score == 4 for item in result.output)


@pytest.mark.parametrize(
    "raw,expected",
    [("4", 4), ("1", 1), ("0", 1), ("-3", 1), ("not-a-number", 1), ("", 1)],
)
def test_parallelism_override_is_read_from_the_environment(monkeypatch, raw, expected):
    if raw:
        monkeypatch.setenv("VOTING_MAX_PARALLEL", raw)
    else:
        monkeypatch.delenv("VOTING_MAX_PARALLEL", raising=False)

    assert max_parallel_requests(OLLAMA) == expected


def test_timeout_is_overridable_for_slow_local_hardware(monkeypatch):
    monkeypatch.setenv("VOTING_TIMEOUT_SECONDS", "600")
    assert default_timeout() == 600.0

    monkeypatch.setenv("VOTING_TIMEOUT_SECONDS", "garbage")
    assert default_timeout() == 120.0


def test_one_judge_limit_covers_concurrent_scoring_runs():
    """UAT generation scores every accepted AC at once. The judge's limit has
    to hold across those runs, not apply separately to each, or the total
    trips the provider's org-wide concurrency limit."""
    judge = JudgeConfig(id="claude-limited", model="anthropic/claude-sonnet-5-5", max_parallel=2)
    client = _RecordingClient(judge)

    async def three_runs_at_once():
        await asyncio.gather(
            *(evaluate_with_combined_model(_input([f"{run}{c}" for c in "abc"]), client) for run in "xyz")
        )

    asyncio.run(three_runs_at_once())

    assert client.peak_in_flight == 2
