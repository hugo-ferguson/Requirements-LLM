"""Batch entry point: score many inputs and save the results as JSON."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voting.models import EvaluationInput, JudgeConfig, load_judges
from voting.voting import evaluate_inputs

DEFAULT_OUTPUT_FILENAME = "votingLayerOutput.json"


def _select_judges(judge_ids: Sequence[str] | None) -> list[JudgeConfig]:
    """Return the enabled judges, narrowed to `judge_ids` if given.

    Ids are case-insensitive. Blank ids are ignored; if all are blank, every
    enabled judge is returned.

    Raises:
        ValueError: If an id doesn't match any enabled judge.
    """
    judges = load_judges()
    if judge_ids is None:
        return judges

    wanted = {
        str(judge_id).strip().lower() for judge_id in judge_ids if str(judge_id).strip()
    }
    known = {judge.id.lower() for judge in judges}
    unknown = wanted - known
    if unknown:
        raise ValueError(
            f"Unknown judge id(s) {sorted(unknown)}; "
            f"config/models.json has {sorted(known)}"
        )
    return [judge for judge in judges if judge.id.lower() in wanted] or judges


async def run_voting_layer(
    inputs: Sequence[dict[str, Any] | EvaluationInput],
    judge_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a batch of generated outputs and save the results.

    Args:
        inputs: `EvaluationInput`s or dicts. Their `judges` are replaced.
        judge_ids: Judges to use. Defaults to every enabled judge.
        output_path: JSON output file. Defaults to votingLayerOutput.json here.

    Returns:
        One dumped `VotingResult` per input, ranked by overall score.

    Raises:
        ValueError: If a judge id doesn't match any enabled judge.
    """
    judges = _select_judges(judge_ids)
    validated_inputs: list[EvaluationInput] = []

    for entry in inputs:
        if isinstance(entry, EvaluationInput):
            validated_inputs.append(entry.model_copy(update={"judges": list(judges)}))
        else:
            validated_inputs.append(
                EvaluationInput.model_validate({**entry, "judges": judges})
            )

    results = await evaluate_inputs(validated_inputs)
    payload = [result.model_dump() for result in results]

    target = (
        Path(output_path)
        if output_path is not None
        else Path(__file__).with_name(DEFAULT_OUTPUT_FILENAME)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def run_voting_layer_sync(
    inputs: Sequence[dict[str, Any] | EvaluationInput],
    judge_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run `run_voting_layer` from synchronous code."""
    return asyncio.run(
        run_voting_layer(inputs, judge_ids=judge_ids, output_path=output_path)
    )
