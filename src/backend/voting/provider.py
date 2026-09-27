from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import litellm

from voting.models import CombinedVote, EvaluationInput, EvaluatedOutput, ProviderFeedback, RubricAverage, RubricFeedback, VotingResult


logger = logging.getLogger(__name__)

RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
COMBINED_PROMPT = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer" / "combined.txt"

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

	Ollama serves one request per model at a time, so firing N candidates
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
	return 1 if model.startswith("ollama/") else 8


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
		prompt = COMBINED_PROMPT.read_text(encoding="utf-8")
		prompt = prompt.replace("{orig_instruction}", instruction)
		prompt = prompt.replace("{orig_response}", response)

		schema_hint = json.dumps(CombinedVote.model_json_schema(), indent=2)

		result = await litellm.acompletion(
			model=self.model,
			messages=[
				{
					"role": "system",
					"content": (
						"You are an acceptance-criteria evaluator. "
						"Return only valid JSON matching this schema:\n"
						f"{schema_hint}"
					),
				},
				{"role": "user", "content": prompt},
			],
			response_format=COMBINED_VOTE_SCHEMA,
			temperature=self.temperature,
			timeout=self.timeout,
			num_retries=self.num_retries,
		)

		content = result.choices[0].message.content
		if not isinstance(content, str) or not content.strip():
			raise ValueError(f"{self.model} returned no message content")

		cleaned = _strip_markdown_fences(content)
		return CombinedVote.model_validate_json(cleaned)


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

	results = await asyncio.gather(
		*(evaluate_output(output) for output in evaluation_input.output),
		return_exceptions=True,
	)
	evaluated_outputs: list[EvaluatedOutput] = []
	failures = 0
	for output, result in zip(evaluation_input.output, results, strict=True):
		if isinstance(result, Exception):
			failures += 1
			# The -1 sentinel below reads as 0.0 once clamped, which is
			# indistinguishable from a candidate the model genuinely rated
			# worthless — and a 0.0 always loses its group. Say so, or a
			# scoring outage quietly decides which candidate wins.
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
		else:
			evaluated_outputs.append(result)

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
