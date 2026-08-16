from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx

from voting.models import (
    EvaluationInput,
    EvaluatedOutput,
    PrometheusVote,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)


RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
RUBRIC_DIR = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer"


class OllamaPrometheusClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("PROMETHEUS_MODEL", "ggozad/prometheus2:latest")
        self.timeout = timeout

    async def evaluate(self, *, instruction: str, response: str, rubric_name: str) -> PrometheusVote:
        rubric = (RUBRIC_DIR / f"{rubric_name}.txt").read_text(encoding="utf-8")
        prompt = rubric.replace("{orig_instruction}", instruction)
        prompt = prompt.replace("{orig_response}", response)

        request = {
            "model": self.model,
            "stream": False,
            "format": PrometheusVote.model_json_schema(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are the Prometheus voting evaluator. Follow the supplied rubric exactly. Return only valid JSON matching the requested schema.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            result = await client.post(f"{self.base_url}/api/chat", json=request)
            result.raise_for_status()

        body: dict[str, Any] = result.json()
        content = body.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama returned no message content")

        return PrometheusVote.model_validate_json(content)


async def evaluate_with_prometheus(evaluation_input: EvaluationInput) -> VotingResult:
    evaluator = OllamaPrometheusClient()

    async def evaluate_output(output: str) -> EvaluatedOutput:
        # Gather rubric evaluations concurrently, but don't fail-fast: collect exceptions
        results = await asyncio.gather(
            *(
                evaluator.evaluate(
                    instruction=evaluation_input.prompt,
                    response=output,
                    rubric_name=rubric_name,
                )
                for rubric_name in RUBRIC_NAMES
            ),
            return_exceptions=True,
        )

        rubric_feedback: list[RubricFeedback] = []
        for rubric_name, res in zip(RUBRIC_NAMES, results, strict=True):
            if isinstance(res, Exception):
                # Convert exceptions into a safe Vote so the caller can continue.
                # Use value -1 to clearly indicate an error occurred.
                error_message = f"Error evaluating rubric {rubric_name}: {res.__class__.__name__}: {res}"
                rubric_feedback.append(RubricFeedback(rubric=rubric_name, value=-1, feedback=error_message))
            else:
                rubric_feedback.append(RubricFeedback(rubric=rubric_name, value=res.score, feedback=res.feedback))

        provider_feedback = ProviderFeedback(
            ai="Prometheus",
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

    # Gather outputs concurrently but tolerate exceptions per-output as well.
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
                ai="Prometheus",
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
