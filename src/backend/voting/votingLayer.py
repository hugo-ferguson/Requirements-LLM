from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from voting.models import EvaluationInput
from voting.voting import COMBINED_PROVIDERS, evaluate_inputs

DEFAULT_PROVIDERS = ["prometheus"] + list(COMBINED_PROVIDERS)
DEFAULT_OUTPUT_FILENAME = "votingLayerOutput.json"


def _normalise_providers(providers: Sequence[str] | None) -> list[str]:
    """Normalise provider selection to lowercase strings and fall back to defaults."""
    if providers is None:
        return list(DEFAULT_PROVIDERS)

    normalised = []
    for provider in providers:
        value = str(provider).strip().lower()
        if value:
            normalised.append(value)

    return normalised or list(DEFAULT_PROVIDERS)


async def run_voting_layer(
    inputs: Sequence[dict[str, Any] | EvaluationInput],
    providers: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a batch of generated outputs and return the saved result payload.

    Args:
        inputs: A list of evaluation items matching the batch format in exampleBatchInput.json.
        providers: List of scoring providers to use, e.g. ["gemini", "prometheus", "qwen"].
        output_path: Optional file path for the JSON output; defaults to a file named
            votingLayerOutput.json next to this module.

    Returns:
        A list of result objects in the same shape as the output produced by exampleCall.py.
    """
    selected_providers = _normalise_providers(providers)
    validated_inputs: list[EvaluationInput] = []

    for item in inputs:
        evaluation_input = item if isinstance(item, EvaluationInput) else EvaluationInput.model_validate(item)
        evaluation_input.providers = list(selected_providers)
        validated_inputs.append(evaluation_input)

    results = await evaluate_inputs(validated_inputs)
    payload = [result.model_dump() for result in results]

    target = Path(output_path) if output_path is not None else Path(__file__).with_name(DEFAULT_OUTPUT_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_voting_layer_sync(
    inputs: Sequence[dict[str, Any] | EvaluationInput],
    providers: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Synchronous convenience wrapper around run_voting_layer."""
    return asyncio.run(run_voting_layer(inputs, providers=providers, output_path=output_path))


if __name__ == "__main__":
    input_path = Path(__file__).with_name("exampleBatchInput.json")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    asyncio.run(run_voting_layer(payload))
