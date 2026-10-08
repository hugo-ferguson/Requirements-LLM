from __future__ import annotations

import asyncio
import logging
import os
import weakref
from pathlib import Path

from pydantic_ai import Agent, RunUsage
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from llm.spec import build_model, with_output_mode
from voting.models import CombinedVote, EvaluationInput, JudgeConfig, EvaluatedOutput, ProviderFeedback, RubricAverage, RubricFeedback, VotingResult


logger = logging.getLogger(__name__)

RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
PROMPT_DIR = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer"
COMBINED_PROMPT = PROMPT_DIR / "combined.txt"
COMBINED_INPUT_PROMPT = PROMPT_DIR / "combined_input.txt"

# One retry of a reply that came back without a usable vote: empty, or not
# matching the schema. Network and rate-limit errors are retried separately,
# by the provider's SDK.
INVALID_REPLY_RETRIES = 1


def default_timeout() -> float:
	"""Per-request timeout, overridable for slow local hardware.

	120s is generous for a cloud model and marginal for a 7B model running on
	CPU, so it has to be tunable without a code change.
	"""
	try:
		return max(1.0, float(os.getenv("VOTING_TIMEOUT_SECONDS", "120")))
	except ValueError:
		return 120.0


def max_parallel_requests(judge: JudgeConfig) -> int:
	"""How many scoring requests may be in flight against `judge` at once.

	`VOTING_MAX_PARALLEL` overrides the judge's `max_parallel`. Ollama serves
	one request per model at a time, so firing N candidates concurrently does
	not make them run in parallel — it queues them, and the per-request
	timeout counts that queue time. Later candidates then expire before they
	ever start, which reads downstream as an unrated candidate rather than as
	a capacity problem. Observed directly: 3 of 5 candidates timing out at
	exactly 120.0s while the first two scored fine.

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
	return judge.max_parallel


# One gate per judge per event loop. The app runs every scoring call on one
# long-lived loop (see app/services/scoring.py), so a judge's `max_parallel`
# caps its requests across all concurrent scoring runs, not just within one.
# Without that, UAT generation scored every accepted AC at once and multiplied
# the cap by the number of ACs, tripping Anthropic's org-wide concurrency
# limit. Keyed by loop because an asyncio primitive binds to the loop that
# first uses it, and tests start a fresh loop each time.
_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]] = (
	weakref.WeakKeyDictionary()
)


def judge_gate(judge: JudgeConfig) -> asyncio.Semaphore:
	"""The running loop's shared concurrency gate for `judge`."""
	gates = _gates.setdefault(asyncio.get_running_loop(), {})
	if judge.id not in gates:
		gates[judge.id] = asyncio.Semaphore(max_parallel_requests(judge))
	return gates[judge.id]


def judge_settings(judge: JudgeConfig) -> ModelSettings:
	"""Request settings for every call `judge` makes.

	`anthropic_cache_instructions` marks the system prompt for caching. Only
	Anthropic reads it; every other provider ignores it.
	"""
	return judge.model_settings(
		timeout=default_timeout(),
		anthropic_cache_instructions=judge.cache_prompt,
	)


def _system_prompt() -> str:
	"""The evaluator role and rubrics, all fixed text.

	Every vote across every story shares this exact prefix. Byte-for-byte
	stability is what lets the cache hit, so nothing per-request belongs here.
	The response schema isn't in it either: the judge's output mode sends it.
	"""
	return (
		"You are an acceptance-criteria evaluator.\n\n"
		f"{COMBINED_PROMPT.read_text(encoding='utf-8')}"
	)


def log_cache_usage(judge: JudgeConfig, usage: RunUsage) -> None:
	"""Debug-log cache reads and writes, the only proof the cache is hitting."""
	if not judge.cache_prompt:
		return
	logger.debug(
		"%s prompt cache: read=%s written=%s input=%s",
		judge.name,
		usage.cache_read_tokens,
		usage.cache_write_tokens,
		usage.input_tokens,
	)


class CombinedJudgeClient:
	"""Scores all four rubrics in one request, for judges with `style: "combined"`."""

	def __init__(self, judge: JudgeConfig, *, model: Model | None = None) -> None:
		self.judge = judge
		self.provider_name = judge.display_name
		self.model = judge.model
		self.agent = Agent(
			model or build_model(judge),
			output_type=with_output_mode(CombinedVote, judge.output_mode),
			instructions=_system_prompt(),
			model_settings=judge_settings(judge),
			retries=INVALID_REPLY_RETRIES,
		)

	async def evaluate(self, *, instruction: str, response: str) -> CombinedVote:
		prompt = COMBINED_INPUT_PROMPT.read_text(encoding="utf-8")
		prompt = prompt.replace("{orig_instruction}", instruction)
		prompt = prompt.replace("{orig_response}", response)

		result = await self.agent.run(prompt)
		log_cache_usage(self.judge, result.usage)
		return result.output


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


async def evaluate_with_combined_model(evaluation_input: EvaluationInput, client: CombinedJudgeClient) -> VotingResult:
	gate = judge_gate(client.judge)

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
	if client.judge.cache_prompt and len(outputs) > 1:
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
