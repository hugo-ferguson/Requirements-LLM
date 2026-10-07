"""A successful voter reply that carries no usable vote.

Regression cover for a live run where Haiku, via LiteLLM, returned empty
message content for 2 of 10 candidates. Each was scored as the -1 sentinel,
lost its group, and showed as 0.0/5 — with nothing logged to say why.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import voting.provider as provider
from voting.models import JudgeConfig
from voting.provider import LiteLLMCombinedClient

VOTE_JSON = json.dumps(
    {
        name: {"feedback": "fine", "score": 4}
        for name in ("correctness", "coverage", "relevance", "understandability")
    }
)


def _reply(content=None, tool_arguments: list[str] | None = None, finish_reason="stop"):
    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name="json_tool_call", arguments=arguments))
        for arguments in (tool_arguments or [])
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


@pytest.fixture
def fake_completion(monkeypatch):
    """Queue replies for litellm.acompletion; returns the list of calls made."""
    replies: list = []
    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return replies.pop(0)

    monkeypatch.setattr(provider.litellm, "acompletion", acompletion)
    return replies, calls


def _evaluate():
    client = LiteLLMCombinedClient(JudgeConfig(id="claude", model="anthropic/claude-haiku-4-5-20251001"))
    return asyncio.run(client.evaluate(instruction="a user story", response="a criterion"))


def test_normal_content_is_parsed_in_one_call(fake_completion) -> None:
    replies, calls = fake_completion
    replies.append(_reply(content=VOTE_JSON))

    vote = _evaluate()

    assert vote.correctness.score == 4
    assert len(calls) == 1
    assert "max_tokens" not in calls[0]


def test_a_judge_without_structured_output_or_temperature_sends_neither(fake_completion) -> None:
    # Live failure: Sonnet 5.5 rejected every vote with "tool_choice: type
    # "tool" and "any" are not supported", because LiteLLM implements
    # response_format for Anthropic as a forced tool call. It also rejects a
    # non-default temperature.
    replies, calls = fake_completion
    replies.append(_reply(content=VOTE_JSON))
    client = LiteLLMCombinedClient(
        JudgeConfig(id="claude", model="anthropic/claude-sonnet-5-5", structured_output=False)
    )

    asyncio.run(client.evaluate(instruction="a user story", response="a criterion"))

    assert "response_format" not in calls[0]
    assert "temperature" not in calls[0]


def test_a_judge_with_structured_output_and_temperature_sends_both(fake_completion) -> None:
    replies, calls = fake_completion
    replies.append(_reply(content=VOTE_JSON))
    client = LiteLLMCombinedClient(JudgeConfig(id="qwen", model="ollama/qwen2.5:7b", temperature=0.1))

    asyncio.run(client.evaluate(instruction="a user story", response="a criterion"))

    assert calls[0]["response_format"] == provider.COMBINED_VOTE_SCHEMA
    assert calls[0]["temperature"] == 0.1


def test_vote_is_recovered_from_tool_calls_when_content_is_empty(fake_completion) -> None:
    replies, calls = fake_completion
    # Two tool calls: the shape LiteLLM does not copy into `content`.
    replies.append(_reply(content=None, tool_arguments=[VOTE_JSON, VOTE_JSON]))

    vote = _evaluate()

    assert vote.relevance.score == 4
    assert len(calls) == 1


def test_an_empty_reply_is_retried_once(fake_completion) -> None:
    replies, calls = fake_completion
    replies.extend([_reply(content=""), _reply(content=VOTE_JSON)])

    vote = _evaluate()

    assert vote.coverage.score == 4
    assert len(calls) == 2


def test_an_unparseable_reply_is_retried_once(fake_completion) -> None:
    replies, calls = fake_completion
    replies.extend([_reply(content="not json"), _reply(content=VOTE_JSON)])

    assert _evaluate().understandability.score == 4
    assert len(calls) == 2


def test_two_empty_replies_raise_with_the_finish_reason(fake_completion) -> None:
    replies, calls = fake_completion
    replies.extend(
        [_reply(content=None, finish_reason="tool_use"), _reply(content=None, finish_reason="tool_use")]
    )

    with pytest.raises(ValueError, match=r"finish_reason='tool_use', tool_calls=0"):
        _evaluate()
    assert len(calls) == 2


def _rubric_call(name: str) -> str:
    return json.dumps({name: {"feedback": f"{name} is fine", "score": 3}})


def test_a_vote_split_into_one_tool_call_per_rubric_is_merged(fake_completion) -> None:
    replies, calls = fake_completion
    # The shape the live failure points to: content empty, one rubric per tool call.
    replies.append(
        _reply(
            content=None,
            tool_arguments=[
                _rubric_call(name)
                for name in ("correctness", "coverage", "relevance", "understandability")
            ],
        )
    )

    vote = _evaluate()

    assert [vote.correctness.score, vote.coverage.score, vote.relevance.score] == [3, 3, 3]
    assert vote.understandability.feedback == "understandability is fine"
    assert len(calls) == 1


def test_an_incomplete_split_vote_is_retried_then_reported(fake_completion) -> None:
    replies, calls = fake_completion
    partial = [_rubric_call("correctness"), _rubric_call("coverage")]
    replies.extend(
        [
            _reply(content=None, tool_arguments=partial, finish_reason="tool_use"),
            _reply(content=None, tool_arguments=partial, finish_reason="tool_use"),
        ]
    )

    with pytest.raises(ValueError, match=r"no complete vote \(finish_reason='tool_use', tool_calls=2\)"):
        _evaluate()
    assert len(calls) == 2


def test_an_openai_reasoning_judge_is_not_sent_temperature_or_max_tokens(fake_completion) -> None:
    # Live failure: gpt-6-luna rejected every vote over `max_tokens`, and
    # reasoning models only accept the default temperature.
    replies, calls = fake_completion
    replies.append(_reply(content=VOTE_JSON))
    client = LiteLLMCombinedClient(JudgeConfig(id="luna", model="openai/gpt-6-luna"))

    asyncio.run(client.evaluate(instruction="a user story", response="a criterion"))

    assert "temperature" not in calls[0]
    assert "max_tokens" not in calls[0]
