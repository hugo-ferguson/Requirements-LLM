from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from voting.models import EvaluationInput, JudgeConfig, load_judges
from voting.voting import evaluate_inputs

DEFAULT_OUTPUT_FILENAME = "votingLayerOutput.json"


def _select_judges(judge_ids: Sequence[str] | None) -> list[JudgeConfig]:
    """The enabled judges from config/models.json, optionally narrowed by id."""
    judges = load_judges()
    if judge_ids is None:
        return judges

    wanted = {str(judge_id).strip().lower() for judge_id in judge_ids if str(judge_id).strip()}
    known = {judge.id.lower() for judge in judges}
    unknown = wanted - known
    if unknown:
        raise ValueError(f"Unknown judge id(s) {sorted(unknown)}; config/models.json has {sorted(known)}")
    return [judge for judge in judges if judge.id.lower() in wanted] or judges


async def run_voting_layer(
    inputs: Sequence[dict[str, Any] | EvaluationInput],
    judge_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a batch of generated outputs and return the saved result payload.

    Args:
        inputs: A list of evaluation items matching the batch format in exampleBatchInput.json.
        judge_ids: Ids of the judges in config/models.json to use, e.g. ["claude"].
            Defaults to every enabled judge.
        output_path: Optional file path for the JSON output; defaults to a file named
            votingLayerOutput.json next to this module.

    Returns:
        A list of result objects in the same shape as the output produced by exampleCall.py.
    """
    judges = _select_judges(judge_ids)
    validated_inputs: list[EvaluationInput] = []

    for item in inputs:
        if isinstance(item, EvaluationInput):
            validated_inputs.append(item.model_copy(update={"judges": list(judges)}))
        else:
            validated_inputs.append(EvaluationInput.model_validate({**item, "judges": judges}))

    results = await evaluate_inputs(validated_inputs)
    payload = [result.model_dump() for result in results]

    target = Path(output_path) if output_path is not None else Path(__file__).with_name(DEFAULT_OUTPUT_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_voting_layer_sync(
    inputs: Sequence[dict[str, Any] | EvaluationInput],
    judge_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Synchronous convenience wrapper around run_voting_layer."""
    return asyncio.run(run_voting_layer(inputs, judge_ids=judge_ids, output_path=output_path))


if __name__ == "__main__":
    input_path = Path(__file__).with_name("exampleBatchInput.json")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    asyncio.run(run_voting_layer(payload))
