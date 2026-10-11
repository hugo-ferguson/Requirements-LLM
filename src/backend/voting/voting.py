from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from voting.models import (
    EvaluationInput,
    EvaluatedOutput,
    JudgeConfig,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)
from voting.prometheus import evaluate_with_prometheus
from voting.provider import CombinedJudgeClient, evaluate_with_combined_model


logger = logging.getLogger(__name__)

RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")

def _provider_error_outputs(evaluation_input: EvaluationInput, provider: str, model: str, error: Exception) -> list[EvaluatedOutput]:
    return [
        EvaluatedOutput(
            output=output,
            feedback=[
                ProviderFeedback(
                    ai=provider,
                    model=model,
                    feedback=[
                        RubricFeedback(
                            rubric=rubric,
                            value=-1,
                            feedback=f"Provider error: {error.__class__.__name__}: {error}",
                        )
                        for rubric in RUBRIC_NAMES
                    ],
                    overall_score=-1.0,
                )
            ],
            rubric_averages=[RubricAverage(
                rubric=rubric, value=-1.0) for rubric in RUBRIC_NAMES],
            overall_score=-1.0,
        )
        for output in evaluation_input.output
    ]


def _merge_provider_outputs(evaluation_input: EvaluationInput, provider_outputs: list[list[EvaluatedOutput]]) -> VotingResult:
    evaluated_outputs: list[EvaluatedOutput] = []
    for output_index, output in enumerate(evaluation_input.output):
        feedback = [outputs[output_index].feedback[0]
                    for outputs in provider_outputs]
        valid_scores = [
            provider.overall_score for provider in feedback if provider.overall_score >= 0]
        overall_score = sum(valid_scores) / \
            len(valid_scores) if valid_scores else -1.0
        rubric_averages = []
        for rubric in RUBRIC_NAMES:
            values = [
                entry.value
                for provider in feedback
                for entry in provider.feedback
                if entry.rubric == rubric and entry.value >= 0
            ]
            rubric_averages.append(
                RubricAverage(rubric=rubric, value=sum(
                    values) / len(values) if values else -1.0)
            )
        evaluated_outputs.append(
            EvaluatedOutput(
                output=output,
                feedback=feedback,
                rubric_averages=rubric_averages,
                overall_score=overall_score,
            )
        )
    evaluated_outputs.sort(key=lambda item: item.overall_score, reverse=True)
    result_overall_score = sum(item.overall_score for item in evaluated_outputs) / \
        len(evaluated_outputs) if evaluated_outputs else -1.0
    return VotingResult(
        ai=evaluation_input.ai,
        model=evaluation_input.model,
        prompt=evaluation_input.prompt,
        output=evaluated_outputs,
        overall_score=result_overall_score,
    )


async def _evaluate_with_judge(judge: JudgeConfig, evaluation_input: EvaluationInput) -> VotingResult:
    if judge.style == "per_rubric":
        return await evaluate_with_prometheus(evaluation_input, judge)
    client = CombinedJudgeClient(judge)
    return await evaluate_with_combined_model(evaluation_input, client)


async def evaluate_input(evaluation_input: EvaluationInput) -> VotingResult:
    judge_results = await asyncio.gather(
        *(_evaluate_with_judge(judge, evaluation_input) for judge in evaluation_input.judges),
        return_exceptions=True,
    )

    provider_outputs: list[list[EvaluatedOutput]] = []
    for result, judge in zip(judge_results, evaluation_input.judges, strict=True):
        if isinstance(result, VotingResult):
            provider_outputs.append(result.output)
        elif isinstance(result, Exception):
            # Degrading to the -1 sentinel is deliberate — a voter outage must
            # not fail generation. But it used to happen silently, so a total
            # misconfiguration returned a successful-looking response full of
            # zeros with nothing logged anywhere. Always say why.
            logger.warning(
                "voting judge %r (model=%s) failed; its rubrics degrade to the "
                "-1 error sentinel, which scores as 0.0: %s: %s",
                judge.id,
                judge.model,
                type(result).__name__,
                result,
            )
            provider_outputs.append(_provider_error_outputs(
                evaluation_input, judge.display_name, judge.model, result))
        else:
            # A BaseException that isn't an Exception (e.g. cancellation) must
            # propagate rather than be scored as a failed judge.
            raise result

    return _merge_provider_outputs(evaluation_input, provider_outputs)


async def evaluate_inputs(evaluation_inputs: list[EvaluationInput]) -> list[VotingResult]:
    """Evaluate every input independently and return them ranked by AI overall score."""
    results = await asyncio.gather(*(evaluate_input(item) for item in evaluation_inputs))
    results.sort(key=lambda item: item.overall_score, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return results


mcp = FastMCP("requirements-voting")


@mcp.tool()
async def evaluate_acceptance_criteria(evaluation_inputs: list[EvaluationInput]) -> list[VotingResult]:
    """Score every input's acceptance criteria against all four voting rubrics."""
    return await evaluate_inputs(evaluation_inputs)


async def main(input_path: str) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        inputs = [EvaluationInput.model_validate(item) for item in payload]
    else:
        inputs = [EvaluationInput.model_validate(payload)]
    results = await evaluate_inputs(inputs)
    print(json.dumps([result.model_dump() for result in results], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated criteria with the voting layer")
    parser.add_argument("input", nargs="?", default=str(
        Path(__file__).with_name("exampleInput.json")))
    parser.add_argument("--mcp", action="store_true",
                        help="Run the MCP server over stdio")
    args = parser.parse_args()
    if args.mcp:
        mcp.run()
    else:
        asyncio.run(main(args.input))
