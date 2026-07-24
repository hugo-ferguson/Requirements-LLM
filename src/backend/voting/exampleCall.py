"""Example caller for the local Prometheus voting evaluator.

Run from src/backend with:

    uv run python -m voting.exampleCall
"""

import asyncio
import json
from pathlib import Path

from voting.voting import EvaluationInput, evaluate_input


async def main() -> None:

	input_path = Path(__file__).with_name("exampleInput.json")
	payload = json.loads(input_path.read_text(encoding="utf-8"))

	# This accepts the same shape as exampleInput.json. Add reference_answer
	# to the JSON when the evaluation has a reference answer.
	evaluation_input = EvaluationInput.model_validate(payload)
	result = await evaluate_input(evaluation_input)

	for vote in result.votes:
		print(f"Output {vote.output_index + 1} - {vote.rubric}: {vote.score}/5")
		print(f"  Criterion: {vote.output}")
		print(f"  Feedback: {vote.feedback}")


if __name__ == "__main__":
	asyncio.run(main())