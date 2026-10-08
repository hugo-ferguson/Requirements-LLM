"""What a judge sends, and what happens when its reply carries no usable vote.

An unusable reply is retried once before the candidate falls back to the -1
sentinel, which loses its group. Each provider gets structured output in a
form it accepts, and none of the settings it rejects.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from tests.fake_llm_api import FakeLlmApi
from voting.models import JudgeConfig
from voting.provider import CombinedJudgeClient

VOTE_JSON = json.dumps(
	{
		name: {"feedback": "fine", "score": 4}
		for name in ("correctness", "coverage", "relevance", "understandability")
	}
)

HAIKU = JudgeConfig(id="claude", provider="anthropic", model="claude-haiku-4-5-20251001")


@pytest.fixture
def api(monkeypatch) -> FakeLlmApi:
	monkeypatch.setenv("FAKE_LLM_API_KEY", "test-key")
	return FakeLlmApi(default_reply=VOTE_JSON)


def _evaluate(api: FakeLlmApi, judge: JudgeConfig = HAIKU):
	client = CombinedJudgeClient(judge, model=api.model(judge))
	return asyncio.run(client.evaluate(instruction="a user story", response="a criterion"))


def test_normal_content_is_parsed_in_one_call(api) -> None:
	vote = _evaluate(api)

	assert vote.correctness.score == 4
	assert len(api.requests) == 1


def test_a_claude_judge_gets_a_json_schema_and_no_tool_call_or_temperature(api) -> None:
	# Sonnet 5.5 rejects a forced tool call and a non-default temperature.
	_evaluate(api, JudgeConfig(id="claude", provider="anthropic", model="claude-sonnet-5-5"))

	[request] = api.requests
	assert request["output_config"]["format"]["type"] == "json_schema"
	assert "tools" not in request
	assert "tool_choice" not in request
	assert "temperature" not in request


def test_an_ollama_judge_gets_a_json_schema_and_its_temperature(api) -> None:
	judge = JudgeConfig(
		id="qwen", provider="ollama", model="qwen2.5:7b", base_url="http://ollama:11434/v1", temperature=0.1
	)

	_evaluate(api, judge)

	[request] = api.requests
	assert request["response_format"]["type"] == "json_schema"
	assert request["temperature"] == 0.1


def test_an_openai_reasoning_judge_is_not_sent_temperature_or_max_tokens(api) -> None:
	# Reasoning models reject `max_tokens` and any non-default temperature.
	_evaluate(api, JudgeConfig(id="luna", provider="openai", model="gpt-6-luna"))

	[request] = api.requests
	assert request["text"]["format"]["type"] == "json_schema"
	assert "temperature" not in request
	assert "max_output_tokens" not in request


def test_an_empty_reply_is_retried_once(api) -> None:
	api.replies.extend(["", VOTE_JSON])

	vote = _evaluate(api)

	assert vote.coverage.score == 4
	assert len(api.requests) == 2


def test_an_unparseable_reply_is_retried_once(api) -> None:
	api.replies.extend(["not json", VOTE_JSON])

	assert _evaluate(api).understandability.score == 4
	assert len(api.requests) == 2


def test_an_incomplete_vote_is_retried_once(api) -> None:
	partial = json.dumps({"correctness": {"feedback": "fine", "score": 3}})
	api.replies.extend([partial, VOTE_JSON])

	assert _evaluate(api).relevance.score == 4
	assert len(api.requests) == 2


def test_two_unusable_replies_raise(api) -> None:
	api.replies.extend(["", ""])

	with pytest.raises(UnexpectedModelBehavior):
		_evaluate(api)
	assert len(api.requests) == 2
