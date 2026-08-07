from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx

from voting.models import EvaluationInput, PrometheusVote, Vote, VotingResult


RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
RUBRIC_DIR = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer"

ERROR_SCORE = 0
DEFAULT_MAX_CONCURRENCY = 4


class OllamaPrometheusClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_concurrency: int | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("PROMETHEUS_MODEL", "ggozad/prometheus2:latest")
        self.timeout = timeout
        self.max_concurrency = max_concurrency or int(
            os.getenv("OLLAMA_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )
        # Bounds this client instance. Share one client across all agents in a
        # run to bound the run as a whole.
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

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

        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                result = await client.post(f"{self.base_url}/api/chat", json=request)
                result.raise_for_status()

        body: dict[str, Any] = result.json()
        content = body.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama returned no message content")

        return PrometheusVote.model_validate_json(content)


async def evaluate_with_prometheus(
    evaluation_input: EvaluationInput,
    evaluator: OllamaPrometheusClient | None = None,
) -> VotingResult:
    evaluator = evaluator or OllamaPrometheusClient()

    async def evaluate_output(output_index: int, output: str) -> list[Vote]:
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

        votes: list[Vote] = []
        for rubric_name, res in zip(RUBRIC_NAMES, results, strict=True):
            if isinstance(res, Exception):
                # Convert exceptions into a safe Vote so the caller can continue.
                feedback = f"Error evaluating rubric {rubric_name}: {res.__class__.__name__}: {res}"
                votes.append(
                    Vote(
                        output_index=output_index,
                        output=output,
                        rubric=rubric_name,
                        feedback=feedback,
                        score=ERROR_SCORE,
                    )
                )
            else:
                votes.append(
                    Vote(
                        output_index=output_index,
                        output=output,
                        rubric=rubric_name,
                        feedback=res.feedback,
                        score=res.score,
                    )
                )

        return votes

    # Gather outputs concurrently but tolerate exceptions per-output as well.
    raw_output_votes = await asyncio.gather(
        *(evaluate_output(index, output) for index, output in enumerate(evaluation_input.output)),
        return_exceptions=True,
    )

    all_votes: list[Vote] = []
    for idx, item in enumerate(raw_output_votes):
        if isinstance(item, Exception):
            # The whole output failed to evaluate; emit one vote per rubric with the error
            for rubric_name in RUBRIC_NAMES:
                feedback = f"Error evaluating output {idx}: {item.__class__.__name__}: {item}"
                all_votes.append(
                    Vote(
                        output_index=idx,
                        output=evaluation_input.output[idx],
                        rubric=rubric_name,
                        feedback=feedback,
                        score=ERROR_SCORE,
                    )
                )
        else:
            # item is a list[Vote]
            all_votes.extend(item)

    return VotingResult(
        ai=evaluation_input.ai,
        model=evaluation_input.model,
        votes=all_votes,
    )
