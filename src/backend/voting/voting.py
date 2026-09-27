from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from voting.models import (
    EvaluationInput,
    EvaluatedOutput,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)
from voting.prometheus import evaluate_with_prometheus
from voting.provider import LiteLLMCombinedClient, evaluate_with_combined_model


logger = logging.getLogger(__name__)

RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")

# `claude` is listed first because app/services/scoring.py asks for that name
# specifically. Without an entry for it the app's only call site resolved to
# "Unknown provider", which _provider_error_outputs turns into the -1 rubric
# sentinel and _scale clamps to 0.0 — i.e. every criterion silently scored
# 0.0/5 with no error anywhere. The name is what scoring.py requires; the
# model behind it is free choice, and a local Ollama model keeps the default
# zero-config and zero-cost.
_DEFAULT_VOTING_PROVIDERS = (
    "claude:ollama/qwen2.5:7b,qwen:ollama/qwen2.5:7b,llama:ollama/llama3.1:8b"
)


def _parse_voting_providers(raw: str) -> dict[str, tuple[str, str]]:
    providers: dict[str, tuple[str, str]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid VOTING_PROVIDERS entry {entry!r} — expected 'name:litellm/model-string'"
            )
        name, model = entry.split(":", 1)
        name = name.strip().lower()
        model = model.strip()
        display_name = name.replace("-", " ").replace("_", " ").title()
        providers[name] = (display_name, model)
    return providers


def _combined_providers() -> dict[str, tuple[str, str]]:
    """
    Resolve VOTING_PROVIDERS on every call rather than once at import.

    The module-level snapshot below is kept for existing importers, but it
    freezes whatever the environment looked like when `voting.voting` was
    first imported — which for the FastAPI app is before most startup work.
    That made a VOTING_PROVIDERS change appear to have no effect until the
    whole process was restarted, and made `.env` look broken (it is: nothing
    here calls load_dotenv, so only real process env vars are ever seen).
    Parsing is a string split over a handful of entries, so per-call cost is
    irrelevant next to the model round-trip that follows.
    """
    return _parse_voting_providers(
        os.getenv("VOTING_PROVIDERS", _DEFAULT_VOTING_PROVIDERS)
    )


# Import-time snapshot, retained because voting.models and voting.votingLayer
# import this name directly. Call sites in this module use _combined_providers().
COMBINED_PROVIDERS = _combined_providers()


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


async def _evaluate_with_provider(name: str, evaluation_input: EvaluationInput) -> VotingResult:
    if name == "prometheus":
        return await evaluate_with_prometheus(evaluation_input)

    combined = _combined_providers()
    if name in combined:
        display_name, model = combined[name]
        client = LiteLLMCombinedClient(provider_name=display_name, model=model)
        return await evaluate_with_combined_model(evaluation_input, client)

    raise ValueError(
        f"Unknown provider: {name!r}. Configured providers are "
        f"{sorted(combined) + ['prometheus']}. Set VOTING_PROVIDERS as a real "
        f"environment variable (a .env entry is not read here) using "
        f"'name:litellm/model-string' pairs."
    )


async def evaluate_input(evaluation_input: EvaluationInput) -> VotingResult:
    combined = _combined_providers()
    selected = [
        (name, combined.get(name, ("Prometheus", os.getenv(
            "PROMETHEUS_MODEL", "ollama/ggozad/prometheus2:latest"))))
        for name in evaluation_input.providers
    ]

    provider_results = await asyncio.gather(
        *(_evaluate_with_provider(name, evaluation_input)
          for name in evaluation_input.providers),
        return_exceptions=True,
    )

    provider_outputs: list[list[EvaluatedOutput]] = []
    for result, (name, (display_name, model)) in zip(provider_results, selected, strict=True):
        if isinstance(result, Exception):
            # Degrading to the -1 sentinel is deliberate — a voter outage must
            # not fail generation. But it used to happen silently, so a total
            # misconfiguration returned a successful-looking response full of
            # zeros with nothing logged anywhere. Always say why.
            logger.warning(
                "voting provider %r (model=%s) failed; its rubrics degrade to the "
                "-1 error sentinel, which scores as 0.0: %s: %s",
                name,
                model,
                type(result).__name__,
                result,
            )
            provider_outputs.append(_provider_error_outputs(
                evaluation_input, display_name, model, result))
        elif isinstance(result, VotingResult):
            provider_outputs.append(result.output)
        else:
            provider_outputs.append(result)

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
