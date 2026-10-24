"""Combined-style judging: one request scores all four rubrics."""

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
from voting.models import (
    CombinedVote,
    EvaluatedOutput,
    EvaluationInput,
    JudgeConfig,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)

logger = logging.getLogger(__name__)

RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
PROMPT_DIR = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer"
COMBINED_PROMPT = PROMPT_DIR / "combined.txt"
COMBINED_INPUT_PROMPT = PROMPT_DIR / "combined_input.txt"

# Retries for an empty or invalid reply. The SDK retries network errors.
INVALID_REPLY_RETRIES = 1


def default_timeout() -> float:
    """Return the per-request timeout from `VOTING_TIMEOUT_SECONDS`, default 120s."""
    try:
        return max(1.0, float(os.getenv("VOTING_TIMEOUT_SECONDS", "120")))
    except ValueError:
        return 120.0


def max_parallel_requests(judge: JudgeConfig) -> int:
    """Return how many concurrent requests `judge` may make.

    A valid `VOTING_MAX_PARALLEL` overrides the judge's `max_parallel`.
    """
    override = os.getenv("VOTING_MAX_PARALLEL")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return judge.max_parallel


# One gate per judge per event loop, so `max_parallel` applies across scoring
# runs. Keyed by loop because tests start a new loop each time.
_gates: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()


def judge_gate(judge: JudgeConfig) -> asyncio.Semaphore:
    """Return the running loop's shared concurrency gate for `judge`."""
    gates = _gates.setdefault(asyncio.get_running_loop(), {})
    if judge.id not in gates:
        gates[judge.id] = asyncio.Semaphore(max_parallel_requests(judge))
    return gates[judge.id]


def judge_settings(judge: JudgeConfig) -> ModelSettings:
    """Return the request settings for every call `judge` makes."""
    return judge.model_settings(
        timeout=default_timeout(),
        anthropic_cache_instructions=judge.cache_prompt,
    )


def _system_prompt() -> str:
    """Return the evaluator role and rubrics.

    Keep per-request text out, so the prompt cache can hit.
    """
    return (
        "You are an acceptance-criteria evaluator.\n\n"
        f"{COMBINED_PROMPT.read_text(encoding='utf-8')}"
    )


def log_cache_usage(judge: JudgeConfig, usage: RunUsage) -> None:
    """Log cache reads and writes, to show whether the cache hit."""
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
        """Build the judge's agent; tests may pass their own `model`."""
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
        """Return the judge's four rubric votes for `response`."""
        prompt = COMBINED_INPUT_PROMPT.read_text(encoding="utf-8")
        prompt = prompt.replace("{orig_instruction}", instruction)
        prompt = prompt.replace("{orig_response}", response)

        result = await self.agent.run(prompt)
        log_cache_usage(self.judge, result.usage)
        return result.output


def error_output(
    output: str, provider_name: str, model: str, message: str
) -> EvaluatedOutput:
    """Return `output` scored -1, the error value, with `message` on every rubric."""
    feedback = [
        RubricFeedback(rubric=rubric_name, value=-1, feedback=message)
        for rubric_name in RUBRIC_NAMES
    ]
    provider_feedback = ProviderFeedback(
        ai=provider_name,
        model=model,
        feedback=feedback,
        overall_score=-1.0,
    )
    return EvaluatedOutput(
        output=output,
        feedback=[provider_feedback],
        rubric_averages=[
            RubricAverage(rubric=rubric_name, value=-1.0)
            for rubric_name in RUBRIC_NAMES
        ],
        overall_score=-1.0,
    )


async def evaluate_with_combined_model(
    evaluation_input: EvaluationInput, client: CombinedJudgeClient
) -> VotingResult:
    """Score every output with one combined judge, one request each.

    A failed request scores -1. Cancellation still propagates.
    """
    gate = judge_gate(client.judge)

    async def evaluate_output(output: str) -> EvaluatedOutput:
        async with gate:
            vote = await client.evaluate(
                instruction=evaluation_input.prompt, response=output
            )
        rubric_feedback = [
            RubricFeedback(
                rubric=rubric_name,
                value=getattr(vote, rubric_name).score,
                feedback=getattr(vote, rubric_name).feedback,
            )
            for rubric_name in RUBRIC_NAMES
        ]
        overall_score = sum(feedback.value for feedback in rubric_feedback) / len(
            rubric_feedback
        )
        provider_feedback = ProviderFeedback(
            ai=client.provider_name,
            model=client.model,
            feedback=rubric_feedback,
            overall_score=overall_score,
        )
        return EvaluatedOutput(
            output=output,
            feedback=[provider_feedback],
            rubric_averages=[
                RubricAverage(rubric=feedback.rubric, value=float(feedback.value))
                for feedback in rubric_feedback
            ],
            overall_score=overall_score,
        )

    outputs = evaluation_input.output
    results: list[EvaluatedOutput | BaseException] = []
    if client.judge.cache_prompt and len(outputs) > 1:
        # Score one candidate first to fill the cache, so the rest can read it.
        results.extend(
            await asyncio.gather(evaluate_output(outputs[0]), return_exceptions=True)
        )
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
                error_output(
                    output,
                    client.provider_name,
                    client.model,
                    f"Error evaluating output: {result.__class__.__name__}: {result}",
                )
            )
        elif isinstance(result, EvaluatedOutput):
            evaluated_outputs.append(result)
        else:
            # Cancellation isn't an Exception, and must propagate.
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
