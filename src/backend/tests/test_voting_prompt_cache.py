"""Prompt caching on the voting layer.

Every vote shares the same rubric and schema text, so for Anthropic judges it
is sent as a cache-marked system block. These tests pin the two things that
silently break caching: per-vote text leaking into the cached prefix, and a
cold batch fired all at once so that nothing is ever read back.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import voting.provider as provider
from voting.models import EvaluationInput
from voting.provider import LiteLLMCombinedClient, evaluate_with_combined_model

VOTE_JSON = json.dumps(
    {
        name: {"feedback": "fine", "score": 4}
        for name in ("correctness", "coverage", "relevance", "understandability")
    }
)


def _capture_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=VOTE_JSON, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    monkeypatch.setattr(provider.litellm, "acompletion", acompletion)
    return calls


def _vote(model: str, response: str) -> None:
    client = LiteLLMCombinedClient("Judge", model)
    asyncio.run(client.evaluate(instruction="a user story", response=response))


def test_anthropic_judges_get_a_cache_marked_system_prefix(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch)

    _vote("anthropic/claude-sonnet-5-5", "a criterion")

    system, user = calls[0]["messages"]
    [block] = system["content"]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert "[Correctness]" in block["text"]
    assert "a user story" not in block["text"]
    assert "a criterion" not in block["text"]
    assert "a user story" in user["content"]
    assert "a criterion" in user["content"]


def test_the_cached_prefix_is_identical_for_every_candidate(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch)

    _vote("anthropic/claude-sonnet-5-5", "first criterion")
    _vote("anthropic/claude-sonnet-5-5", "second criterion")

    assert calls[0]["messages"][0] == calls[1]["messages"][0]


def test_other_judges_get_a_plain_system_message(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch)

    _vote("openai/gpt-6-luna", "a criterion")

    system = calls[0]["messages"][0]
    assert isinstance(system["content"], str)
    assert "[Correctness]" in system["content"]


class _RecordingClient:
    """Stands in for LiteLLMCombinedClient, recording which votes overlap."""

    def __init__(self, model: str) -> None:
        self.provider_name = "Test"
        self.model = model
        self.in_flight: set[str] = set()
        self.overlaps: dict[str, set[str]] = {}

    async def evaluate(self, *, instruction: str, response: str):
        self.overlaps[response] = set(self.in_flight)
        for other in self.in_flight:
            self.overlaps[other].add(response)
        self.in_flight.add(response)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.in_flight.discard(response)

        rubric = SimpleNamespace(score=4, feedback="fine")
        return SimpleNamespace(
            correctness=rubric, coverage=rubric, relevance=rubric, understandability=rubric
        )


def _input(candidates: list[str]) -> EvaluationInput:
    return EvaluationInput(
        ai="requirements-llm",
        model="test",
        prompt="a user story",
        output=candidates,
        reference_answer="",
        providers=["claude"],
    )


def test_an_anthropic_batch_scores_one_candidate_before_the_rest() -> None:
    client = _RecordingClient("anthropic/claude-sonnet-5-5")

    result = asyncio.run(evaluate_with_combined_model(_input(["a", "b", "c", "d"]), client))

    assert [item.output for item in result.output] == ["a", "b", "c", "d"]
    assert client.overlaps["a"] == set(), "the first vote must finish before others start"
    assert client.overlaps["b"] >= {"c", "d"}, "the rest still run concurrently"
