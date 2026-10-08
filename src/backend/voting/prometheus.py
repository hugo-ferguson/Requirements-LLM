from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model

from llm.spec import build_model, with_output_mode
from voting.provider import INVALID_REPLY_RETRIES, judge_gate, judge_settings, log_cache_usage
from voting.models import (
    EvaluationInput,
    EvaluatedOutput,
    JudgeConfig,
    PrometheusVote,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)


RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
RUBRIC_DIR = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer"
SYSTEM_PROMPT = "You are the Prometheus voting evaluator. Follow the supplied rubric exactly."


class PrometheusJudgeClient:
    """Scores one rubric per call, for judges with `style: "per_rubric"`."""

    def __init__(self, judge: JudgeConfig, *, model: Model | None = None) -> None:
        self.judge = judge
        self.model = judge.model
        self.agent = Agent(
            model or build_model(judge),
            output_type=with_output_mode(PrometheusVote, judge.output_mode),
            instructions=SYSTEM_PROMPT,
            model_settings=judge_settings(judge),
            retries=INVALID_REPLY_RETRIES,
        )

    async def evaluate(self, *, instruction: str, response: str, rubric_name: str) -> PrometheusVote:
        rubric = (RUBRIC_DIR / f"{rubric_name}.txt").read_text(encoding="utf-8")
        prompt = rubric.replace("{orig_instruction}", instruction)
        prompt = prompt.replace("{orig_response}", response)

        result = await self.agent.run(prompt)
        log_cache_usage(self.judge, result.usage)
        return result.output


async def evaluate_with_prometheus(evaluation_input: EvaluationInput, judge: JudgeConfig) -> VotingResult:
    evaluator = PrometheusJudgeClient(judge)
    # Same queue-timeout trap as the combined path, and worse here: this fans
    # out four rubric calls per candidate, so an unbounded gather over N
    # candidates puts 4N requests in front of a server that runs one at a
    # time. See `max_parallel_requests` for why serialising costs nothing.
    gate = judge_gate(judge)

    async def evaluate_output(output: str) -> EvaluatedOutput:
        async def scored(rubric_name: str) -> PrometheusVote:
            async with gate:
                return await evaluator.evaluate(
                    instruction=evaluation_input.prompt,
                    response=output,
                    rubric_name=rubric_name,
                )

        results = await asyncio.gather(
            *(scored(rubric_name) for rubric_name in RUBRIC_NAMES),
            return_exceptions=True,
        )

        rubric_feedback: list[RubricFeedback] = []
        for rubric_name, res in zip(RUBRIC_NAMES, results, strict=True):
            if isinstance(res, Exception):
                error_message = f"Error evaluating rubric {rubric_name}: {res.__class__.__name__}: {res}"
                rubric_feedback.append(RubricFeedback(rubric=rubric_name, value=-1, feedback=error_message))
            else:
                rubric_feedback.append(RubricFeedback(rubric=rubric_name, value=res.score, feedback=res.feedback))

        provider_feedback = ProviderFeedback(
            ai=judge.display_name,
            model=evaluator.model,
            feedback=rubric_feedback,
            overall_score=sum(item.value for item in rubric_feedback) / len(rubric_feedback),
        )
        rubric_averages = [RubricAverage(rubric=item.rubric, value=float(item.value)) for item in rubric_feedback]
        return EvaluatedOutput(
            output=output,
            feedback=[provider_feedback],
            rubric_averages=rubric_averages,
            overall_score=provider_feedback.overall_score,
        )

    raw_output_votes = await asyncio.gather(
        *(evaluate_output(output) for output in evaluation_input.output),
        return_exceptions=True,
    )

    evaluated_outputs: list[EvaluatedOutput] = []
    for idx, item in enumerate(raw_output_votes):
        if isinstance(item, Exception):
            error_feedback = [
                RubricFeedback(
                    rubric=rubric_name,
                    value=-1,
                    feedback=f"Error evaluating output {idx}: {item.__class__.__name__}: {item}",
                )
                for rubric_name in RUBRIC_NAMES
            ]
            provider_feedback = ProviderFeedback(
                ai=judge.display_name,
                model=evaluator.model,
                feedback=error_feedback,
                overall_score=-1.0,
            )
            evaluated_outputs.append(
                EvaluatedOutput(
                    output=evaluation_input.output[idx],
                    feedback=[provider_feedback],
                    rubric_averages=[RubricAverage(rubric=name, value=-1.0) for name in RUBRIC_NAMES],
                    overall_score=-1.0,
                )
            )
        else:
            evaluated_outputs.append(item)

    return VotingResult(
        ai=evaluation_input.ai,
        model=evaluation_input.model,
        prompt=evaluation_input.prompt,
        output=evaluated_outputs,
    )
