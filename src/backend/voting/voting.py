"""Run every judge over an input and merge their votes.

Run `python -m voting.voting` from src/backend to serve it as an MCP tool.
"""

from __future__ import annotations

import asyncio
import logging

from mcp.server.fastmcp import FastMCP

from voting.models import (
    EvaluatedOutput,
    EvaluationInput,
    JudgeConfig,
    RubricAverage,
    VotingResult,
)
from voting.prometheus import evaluate_with_prometheus
from voting.provider import (
    RUBRIC_NAMES,
    CombinedJudgeClient,
    error_output,
    evaluate_with_combined_model,
)

logger = logging.getLogger(__name__)


def _provider_error_outputs(
    evaluation_input: EvaluationInput,
    provider_name: str,
    model: str,
    error: Exception,
) -> list[EvaluatedOutput]:
    """Return -1 on every rubric of every output, for a failed judge."""
    message = f"Provider error: {error.__class__.__name__}: {error}"
    return [
        error_output(output, provider_name, model, message)
        for output in evaluation_input.output
    ]


def _merge_provider_outputs(
    evaluation_input: EvaluationInput, judge_outputs: list[list[EvaluatedOutput]]
) -> VotingResult:
    """Average every judge's scores per output, ignoring -1, and rank best first.

    Each judge's list must be in the order of `evaluation_input.output`.
    """
    evaluated_outputs: list[EvaluatedOutput] = []
    for output_index, output in enumerate(evaluation_input.output):
        judge_feedback = [
            outputs[output_index].feedback[0] for outputs in judge_outputs
        ]
        valid_scores = [
            feedback.overall_score
            for feedback in judge_feedback
            if feedback.overall_score >= 0
        ]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else -1.0
        rubric_averages = []
        for rubric in RUBRIC_NAMES:
            values = [
                rubric_feedback.value
                for feedback in judge_feedback
                for rubric_feedback in feedback.feedback
                if rubric_feedback.rubric == rubric and rubric_feedback.value >= 0
            ]
            rubric_averages.append(
                RubricAverage(
                    rubric=rubric, value=sum(values) / len(values) if values else -1.0
                )
            )
        evaluated_outputs.append(
            EvaluatedOutput(
                output=output,
                feedback=judge_feedback,
                rubric_averages=rubric_averages,
                overall_score=overall_score,
            )
        )
    evaluated_outputs.sort(key=lambda evaluated: evaluated.overall_score, reverse=True)
    result_overall_score = (
        sum(evaluated.overall_score for evaluated in evaluated_outputs)
        / len(evaluated_outputs)
        if evaluated_outputs
        else -1.0
    )
    return VotingResult(
        ai=evaluation_input.ai,
        model=evaluation_input.model,
        prompt=evaluation_input.prompt,
        output=evaluated_outputs,
        overall_score=result_overall_score,
    )


async def _evaluate_with_judge(
    judge: JudgeConfig, evaluation_input: EvaluationInput
) -> VotingResult:
    """Score every output with one judge, in the judge's voting style."""
    if judge.style == "per_rubric":
        return await evaluate_with_prometheus(evaluation_input, judge)
    client = CombinedJudgeClient(judge)
    return await evaluate_with_combined_model(evaluation_input, client)


async def evaluate_input(evaluation_input: EvaluationInput) -> VotingResult:
    """Score one input with every judge at once and merge the votes.

    A failed judge scores -1 on every rubric. Cancellation still propagates.
    """
    judge_results = await asyncio.gather(
        *(
            _evaluate_with_judge(judge, evaluation_input)
            for judge in evaluation_input.judges
        ),
        return_exceptions=True,
    )

    judge_outputs: list[list[EvaluatedOutput]] = []
    for result, judge in zip(judge_results, evaluation_input.judges, strict=True):
        if isinstance(result, VotingResult):
            judge_outputs.append(result.output)
        elif isinstance(result, Exception):
            # Log it, or a broken config looks like a run full of zero scores.
            logger.warning(
                "voting judge %r (model=%s) failed; its rubrics degrade to the "
                "-1 error sentinel, which scores as 0.0: %s: %s",
                judge.id,
                judge.model,
                type(result).__name__,
                result,
            )
            judge_outputs.append(
                _provider_error_outputs(
                    evaluation_input, judge.display_name, judge.model, result
                )
            )
        else:
            # Cancellation isn't an Exception, and must propagate.
            raise result

    return _merge_provider_outputs(evaluation_input, judge_outputs)


async def evaluate_inputs(
    evaluation_inputs: list[EvaluationInput],
) -> list[VotingResult]:
    """Score each input and return the results ranked best first."""
    results = await asyncio.gather(
        *(evaluate_input(evaluation_input) for evaluation_input in evaluation_inputs)
    )
    results.sort(key=lambda result: result.overall_score, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return results


mcp = FastMCP("requirements-voting")


@mcp.tool()
async def evaluate_acceptance_criteria(
    evaluation_inputs: list[EvaluationInput],
) -> list[VotingResult]:
    """Score each input's acceptance criteria on all four rubrics."""
    return await evaluate_inputs(evaluation_inputs)


if __name__ == "__main__":
    mcp.run()
