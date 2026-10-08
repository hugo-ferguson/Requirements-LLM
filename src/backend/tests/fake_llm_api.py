"""A stand-in for the providers' HTTP APIs, for tests that check what is sent.

`FakeLlmApi.model(spec)` builds the real PydanticAI model for a model entry,
with a mock transport underneath, so the request each provider would receive
can be inspected without a network or an API key. Replies are queued as text
and answered in whichever API's shape the request was for.
"""

from __future__ import annotations

import json

import httpx2

from llm.spec import ModelSpec, build_model


class FakeLlmApi:
	def __init__(self, default_reply: str = "") -> None:
		self.default_reply = default_reply
		self.replies: list[str] = []
		self.requests: list[dict] = []

	def model(self, spec: ModelSpec):
		if spec.api_key_env is None and spec.provider != "ollama":
			spec = spec.model_copy(update={"api_key_env": "FAKE_LLM_API_KEY"})
		return build_model(spec, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._reply)))

	def _reply(self, request: httpx2.Request) -> httpx2.Response:
		self.requests.append(json.loads(request.content))
		text = self.replies.pop(0) if self.replies else self.default_reply
		path = request.url.path
		if path.endswith("/messages"):
			return httpx2.Response(200, json=_anthropic(text))
		if path.endswith("/responses"):
			return httpx2.Response(200, json=_openai_responses(text))
		return httpx2.Response(200, json=_chat_completions(text))


def _anthropic(text: str) -> dict:
	return {
		"id": "msg_test",
		"type": "message",
		"role": "assistant",
		"model": "test",
		"content": [{"type": "text", "text": text}],
		"stop_reason": "end_turn",
		"stop_sequence": None,
		"usage": {"input_tokens": 1, "output_tokens": 1},
	}


def _openai_responses(text: str) -> dict:
	return {
		"id": "resp_test",
		"object": "response",
		"created_at": 0,
		"status": "completed",
		"model": "test",
		"output": [
			{
				"type": "message",
				"id": "msg_test",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": text, "annotations": []}],
			}
		],
		"parallel_tool_calls": False,
		"tool_choice": "auto",
		"tools": [],
		"usage": {
			"input_tokens": 1,
			"output_tokens": 1,
			"total_tokens": 2,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0},
		},
	}


def _chat_completions(text: str) -> dict:
	return {
		"id": "chatcmpl_test",
		"object": "chat.completion",
		"created": 0,
		"model": "test",
		"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
		"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
	}
