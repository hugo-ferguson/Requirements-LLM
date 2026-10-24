"""Prometheus-style judging: one request per rubric."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.models import Model

from llm.spec import build_model, with_output_mode
from voting.models import (
    EvaluatedOutput,
    EvaluationInput,
    JudgeConfig,
    PrometheusVote,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)
from voting.provider import (
    INVALID_REPLY_RETRIES,
    PROMPT_DIR,
    RUBRIC_NAMES,
    error_output,
    judge_gate,
    judge_settings,
    log_cache_usage,
)

SYSTEM_PROMPT = (
    "You are the Prometheus voting evaluator. Follow the supplied rubric exactly."
)


class PrometheusJudgeClient:
    """Scores one rubric per call, for judges with `style: "per_rubric"`."""

    def __init__(self, judge: JudgeConfig, *, model: Model | None = None) -> None:
        """Build the judge's agent; tests may pass their own `model`."""
        self.judge = judge
        self.model = judge.model
        self.agent = Agent(
            model or build_model(judge),
            output_type=with_output_mode(PrometheusVote, judge.output_mode),
            instructions=SYSTEM_PROMPT,
            model_settings=judge_settings(judge),
            retries=INVALID_REPLY_RETRIES,
        )

    async def evaluate(
        self, *, instruction: str, response: str, rubric_name: str
    ) -> PrometheusVote:
        """Return the judge's vote on one rubric for `response`."""
        template = (PROMPT_DIR / f"{rubric_name}.txt").read_text(encoding="utf-8")
        prompt = template.replace("{orig_instruction}", instruction)
        prompt = prompt.replace("{orig_response}", response)

        result = await self.agent.run(prompt)
        log_cache_usage(self.judge, result.usage)
        return result.output


async def evaluate_with_prometheus(
    evaluation_input: EvaluationInput, judge: JudgeConfig
) -> VotingResult:
    """Score every output with one per-rubric judge, four calls each.

    A failed call scores -1 on its rubric. Cancellation still propagates.
    """
    evaluator = PrometheusJudgeClient(judge)
    # Limits concurrent calls, so queued requests don't time out.
    gate = judge_gate(judge)

    async def evaluate_output(output: str) -> EvaluatedOutput:
        async def score_rubric(rubric_name: str) -> PrometheusVote:
            async with gate:
                return await evaluator.evaluate(
                    instruction=evaluation_input.prompt,
                    response=output,
                    rubric_name=rubric_name,
                )

        votes = await asyncio.gather(
            *(score_rubric(rubric_name) for rubric_name in RUBRIC_NAMES),
            return_exceptions=True,
        )

        rubric_feedback: list[RubricFeedback] = []
        for rubric_name, vote in zip(RUBRIC_NAMES, votes, strict=True):
            if isinstance(vote, Exception):
                error_message = (
                    f"Error evaluating rubric {rubric_name}: "
                    f"{vote.__class__.__name__}: {vote}"
                )
                rubric_feedback.append(
                    RubricFeedback(rubric=rubric_name, value=-1, feedback=error_message)
                )
            elif isinstance(vote, BaseException):
                # Cancellation isn't an Exception, and must propagate.
                raise vote
            else:
                rubric_feedback.append(
                    RubricFeedback(
                        rubric=rubric_name, value=vote.score, feedback=vote.feedback
                    )
                )

        provider_feedback = ProviderFeedback(
            ai=judge.display_name,
            model=evaluator.model,
            feedback=rubric_feedback,
            overall_score=sum(feedback.value for feedback in rubric_feedback)
            / len(rubric_feedback),
        )
        rubric_averages = [
            RubricAverage(rubric=feedback.rubric, value=float(feedback.value))
            for feedback in rubric_feedback
        ]
        return EvaluatedOutput(
            output=output,
            feedback=[provider_feedback],
            rubric_averages=rubric_averages,
            overall_score=provider_feedback.overall_score,
        )

    results = await asyncio.gather(
        *(evaluate_output(output) for output in evaluation_input.output),
        return_exceptions=True,
    )

    evaluated_outputs: list[EvaluatedOutput] = []
    for index, (output, result) in enumerate(
        zip(evaluation_input.output, results, strict=True)
    ):
        if isinstance(result, Exception):
            evaluated_outputs.append(
                error_output(
                    output,
                    judge.display_name,
                    evaluator.model,
                    f"Error evaluating output {index}: "
                    f"{result.__class__.__name__}: {result}",
                )
            )
        elif isinstance(result, BaseException):
            raise result
        else:
            evaluated_outputs.append(result)

    return VotingResult(
        ai=evaluation_input.ai,
        model=evaluation_input.model,
        prompt=evaluation_input.prompt,
        output=evaluated_outputs,
    )
