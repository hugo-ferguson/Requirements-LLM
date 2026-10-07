from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import litellm
from pydantic import ValidationError

from voting.models import CombinedVote, EvaluationInput, EvaluatedOutput, ProviderFeedback, RubricAverage, RubricFeedback, VotingResult


logger = logging.getLogger(__name__)

RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
PROMPT_DIR = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer"
COMBINED_PROMPT = PROMPT_DIR / "combined.txt"
COMBINED_INPUT_PROMPT = PROMPT_DIR / "combined_input.txt"

# Two attempts: one retry of a reply that came back without a usable vote.
EMPTY_REPLY_ATTEMPTS = 2

COMBINED_VOTE_SCHEMA = {
	"type": "json_schema",
	"json_schema": {
		"name": "CombinedVote",
		"schema": CombinedVote.model_json_schema(),
	},
}


def _strip_markdown_fences(text: str) -> str:
	"""Remove markdown code fences that some models wrap JSON output in."""
	stripped = text.strip()
	if stripped.startswith("```"):
		stripped = stripped.strip("`")
		if stripped.lower().startswith("json"):
			stripped = stripped[4:].lstrip()
	match = re.search(r"\{.*\}", stripped, re.DOTALL)
	if match:
		return match.group(0)
	return stripped


def _merge_json_objects(texts: list[str]) -> dict:
	"""Combine JSON objects spread over several strings; later keys win."""
	merged: dict = {}
	for text in texts:
		try:
			value = json.loads(_strip_markdown_fences(text))
		except json.JSONDecodeError:
			continue
		if isinstance(value, dict):
			merged.update(value)
	return merged


@dataclass(frozen=True)
class JudgeProfile:
	"""How a judge's provider differs from the LiteLLM defaults.

	LiteLLM's own capability data can't be trusted for these: it reports
	Sonnet 5.5 as supporting `tool_choice` and `temperature`, and the API
	rejects both. So the differences are declared here, per provider.
	"""

	# Send `response_format`. LiteLLM implements it for Anthropic as a forced
	# tool call, which Claude from Sonnet 5.5 / Opus 5.5 on rejects with a 400.
	# Without it, the schema in the system prompt and `_parse_vote` suffice.
	structured_output: bool = True
	# "never" for providers whose current models reject a non-default value
	# (Claude from Opus 4.7 / Sonnet 5 on); "non_reasoning" when only
	# reasoning models do, as LiteLLM can tell for OpenAI.
	temperature: Literal["always", "never", "non_reasoning"] = "always"
	# Mark the shared prefix with `cache_control`. Only Anthropic needs it:
	# OpenAI and Gemini cache long prefixes on their own, and LiteLLM may not
	# strip the marker for every provider.
	cache_marker: bool = False
	# Concurrent requests. Ollama serves one request per model at a time, so
	# extra concurrency only queues, and the queue time counts against each
	# request's timeout (see `max_parallel_requests`).
	max_parallel: int = 8


JUDGE_PROFILES: dict[str, JudgeProfile] = {
	"anthropic": JudgeProfile(structured_output=False, temperature="never", cache_marker=True),
	"openai": JudgeProfile(temperature="non_reasoning"),
	"ollama": JudgeProfile(max_parallel=1),
}


def judge_profile(model: str) -> JudgeProfile:
	"""The profile for a LiteLLM model string, keyed on its `provider/` prefix."""
	provider, _, _ = model.partition("/")
	return JUDGE_PROFILES.get(provider, JudgeProfile())


def _accepts_temperature(model: str) -> bool:
	"""False when the request would be rejected for a non-default temperature."""
	rule = judge_profile(model).temperature
	if rule != "non_reasoning":
		return rule == "always"
	try:
		return not litellm.supports_reasoning(model=model)
	except Exception:
		return True


def default_timeout() -> float:
	"""Per-request timeout, overridable for slow local hardware.

	120s is generous for a cloud model and marginal for a 7B model running on
	CPU, so it has to be tunable without a code change.
	"""
	try:
		return max(1.0, float(os.getenv("VOTING_TIMEOUT_SECONDS", "120")))
	except ValueError:
		return 120.0


def max_parallel_requests(model: str) -> int:
	"""How many scoring requests may be in flight against `model` at once.

	`VOTING_MAX_PARALLEL` overrides the provider's profile. Ollama serves one request per model at a time, so firing N candidates
	concurrently does not make them run in parallel — it queues them, and the
	per-request timeout counts that queue time. Later candidates then expire
	before they ever start, which reads downstream as an unrated candidate
	rather than as a capacity problem. Observed directly: 3 of 5 candidates
	timing out at exactly 120.0s while the first two scored fine.

	Serialising costs nothing in wall-clock terms, because Ollama was going to
	run them one at a time regardless. It just means each timeout covers real
	work instead of waiting in line.
	"""
	raw = os.getenv("VOTING_MAX_PARALLEL")
	if raw:
		try:
			return max(1, int(raw))
		except ValueError:
			pass
	return judge_profile(model).max_parallel


def _system_message(model: str) -> dict:
	"""The evaluator role, response schema and rubrics, all fixed text.

	Every vote across every story shares this exact prefix. Byte-for-byte
	stability is what lets the cache hit, so nothing per-request belongs here.
	"""
	schema_hint = json.dumps(CombinedVote.model_json_schema(), indent=2)
	text = (
		"You are an acceptance-criteria evaluator. "
		"Return only valid JSON matching this schema:\n"
		f"{schema_hint}\n\n"
		f"{COMBINED_PROMPT.read_text(encoding='utf-8')}"
	)
	if not judge_profile(model).cache_marker:
		return {"role": "system", "content": text}
	return {
		"role": "system",
		"content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}],
	}


def _log_cache_usage(model: str, result) -> None:
	"""Debug-log cache reads and writes, the only proof the cache is hitting."""
	usage = getattr(result, "usage", None)
	if usage is None or not judge_profile(model).cache_marker:
		return
	logger.debug(
		"%s prompt cache: read=%s written=%s uncached=%s",
		model,
		getattr(usage, "cache_read_input_tokens", None),
		getattr(usage, "cache_creation_input_tokens", None),
		getattr(usage, "prompt_tokens", None),
	)


class LiteLLMCombinedClient:
	"""Evaluates acceptance criteria using any LiteLLM-supported model."""

	def __init__(
		self,
		provider_name: str,
		model: str,
		*,
		temperature: float = 0.1,
		timeout: float | None = None,
		num_retries: int = 2,
	) -> None:
		timeout = default_timeout() if timeout is None else timeout
		self.provider_name = provider_name
		self.model = model
		self.temperature = temperature
		self.timeout = timeout
		self.num_retries = num_retries

	async def evaluate(self, *, instruction: str, response: str) -> CombinedVote:
		prompt = COMBINED_INPUT_PROMPT.read_text(encoding="utf-8")
		prompt = prompt.replace("{orig_instruction}", instruction)
		prompt = prompt.replace("{orig_response}", response)

		# No max_tokens: OpenAI reasoning models reject it (LiteLLM doesn't
		# translate it to max_completion_tokens for them), and a small cap would
		# eat into their reasoning tokens and truncate the vote.
		sampling = {"temperature": self.temperature} if _accepts_temperature(self.model) else {}

		structured = (
			{"response_format": COMBINED_VOTE_SCHEMA}
			if judge_profile(self.model).structured_output
			else {}
		)

		# A successful reply can still carry no usable vote, and LiteLLM's own
		# num_retries only covers HTTP/network errors — so retry that case here.
		last_error: Exception | None = None
		for _ in range(EMPTY_REPLY_ATTEMPTS):
			result = await litellm.acompletion(
				model=self.model,
				messages=[
					_system_message(self.model),
					{"role": "user", "content": prompt},
				],
				**structured,
				**sampling,
				# Drop whatever else a given judge doesn't support instead of
				# failing the vote.
				drop_params=True,
				timeout=self.timeout,
				num_retries=self.num_retries,
			)
			_log_cache_usage(self.model, result)
			try:
				return self._parse_vote(result.choices[0])
			except (ValueError, ValidationError) as error:
				last_error = error

		assert last_error is not None
		raise last_error

	def _parse_vote(self, choice) -> CombinedVote:
		"""Read the vote from the reply text, falling back to its tool calls.

		Where LiteLLM implements `response_format` as a tool call (it did for
		Anthropic, before Anthropic judges stopped receiving it), it only copies
		the arguments into `content` when the reply holds exactly one call to
		its JSON tool. A model can split the vote into one tool call per
		rubric instead, which leaves `content` empty —
		how candidates ended up unrated with "returned no message content". So
		the tool calls are merged back into a single vote.
		"""
		message = choice.message
		tool_calls = getattr(message, "tool_calls", None) or []
		shape = (
			f"finish_reason={getattr(choice, 'finish_reason', None)!r}, "
			f"tool_calls={len(tool_calls)}"
		)

		arguments = [
			call.function.arguments
			for call in tool_calls
			if isinstance(getattr(call.function, "arguments", None), str)
		]
		texts = [message.content] if isinstance(message.content, str) else []
		merged = _merge_json_objects(arguments)
		if merged:
			texts.append(json.dumps(merged))
		texts += arguments
		texts = [text for text in texts if text.strip()]

		if not texts:
			raise ValueError(f"{self.model} returned no message content ({shape})")

		first_error: ValidationError | None = None
		for text in texts:
			try:
				return CombinedVote.model_validate_json(_strip_markdown_fences(text))
			except ValidationError as error:
				first_error = first_error or error
		raise ValueError(
			f"{self.model} returned no complete vote ({shape}): {first_error}"
		) from first_error


def _error_output(output: str, provider_name: str, model: str, message: str) -> EvaluatedOutput:
	feedback = [RubricFeedback(rubric=name, value=-1, feedback=message) for name in RUBRIC_NAMES]
	provider_feedback = ProviderFeedback(
		ai=provider_name,
		model=model,
		feedback=feedback,
		overall_score=-1.0,
	)
	return EvaluatedOutput(
		output=output,
		feedback=[provider_feedback],
		rubric_averages=[RubricAverage(rubric=name, value=-1.0) for name in RUBRIC_NAMES],
		overall_score=-1.0,
	)


async def evaluate_with_combined_model(evaluation_input: EvaluationInput, client: LiteLLMCombinedClient) -> VotingResult:
	# Created per call rather than per module: an asyncio primitive binds to
	# the running loop, and score_candidates starts a fresh loop every time.
	gate = asyncio.Semaphore(max_parallel_requests(client.model))

	async def evaluate_output(output: str) -> EvaluatedOutput:
		async with gate:
			result = await client.evaluate(instruction=evaluation_input.prompt, response=output)
		rubric_feedback = [
			RubricFeedback(
				rubric=name,
				value=getattr(result, name).score,
				feedback=getattr(result, name).feedback,
			)
			for name in RUBRIC_NAMES
		]
		overall_score = sum(item.value for item in rubric_feedback) / len(rubric_feedback)
		provider_feedback = ProviderFeedback(
			ai=client.provider_name,
			model=client.model,
			feedback=rubric_feedback,
			overall_score=overall_score,
		)
		return EvaluatedOutput(
			output=output,
			feedback=[provider_feedback],
			rubric_averages=[RubricAverage(rubric=item.rubric, value=float(item.value)) for item in rubric_feedback],
			overall_score=overall_score,
		)

	outputs = evaluation_input.output
	results: list[EvaluatedOutput | BaseException] = []
	if judge_profile(client.model).cache_marker and len(outputs) > 1:
		# A cache entry only exists once the first request has started
		# responding, so a cold batch fired all at once would pay the write
		# premium on every vote and read nothing. Score one candidate first to
		# write the shared rubric prefix, then the rest read it.
		results.extend(await asyncio.gather(evaluate_output(outputs[0]), return_exceptions=True))
		outputs = outputs[1:]
	results.extend(
		await asyncio.gather(
			*(evaluate_output(output) for output in outputs),
			return_exceptions=True,
		)
	)
	evaluated_outputs: list[EvaluatedOutput] = []
	failures = 0
	for output, result in zip(evaluation_input.output, results, strict=True):
		if isinstance(result, Exception):
			failures += 1
			logger.warning(
				"%s (model=%s) failed to score a candidate; it degrades to the "
				"-1 sentinel and will rank last: %s: %s",
				client.provider_name,
				client.model,
				result.__class__.__name__,
				result,
			)
			evaluated_outputs.append(
				_error_output(
					output,
					client.provider_name,
					client.model,
					f"Error evaluating output: {result.__class__.__name__}: {result}",
				)
			)
		elif isinstance(result, EvaluatedOutput):
			evaluated_outputs.append(result)
		else:
			raise result

	if failures:
		logger.warning(
			"%s scored %d of %d candidate(s); %d failed and will rank last",
			client.provider_name,
			len(evaluated_outputs) - failures,
			len(evaluated_outputs),
			failures,
		)
	return VotingResult(
		ai=evaluation_input.ai,
		model=evaluation_input.model,
		prompt=evaluation_input.prompt,
		output=evaluated_outputs,
	)
