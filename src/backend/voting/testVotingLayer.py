"""Small runner for testing the standalone voting layer.

Run from src/backend with:

    uv run python -m voting.testVotingLayer

This reads the example batch input, calls the reusable voting layer, prints the
returned evaluation data in a human-readable format, and leaves the last result
saved to votingLayerOutput.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from voting.votingLayer import run_voting_layer


async def main() -> None:
    input_path = Path(__file__).with_name("exampleBatchInput.json")
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    results = await run_voting_layer(payload)

    for result in results:
        print(f"AI: {result['ai']} | Model: {result['model']}")
        for evaluated_output in result["output"]:
            print(f"Output: {evaluated_output['output']}")
            print(f"Overall score: {evaluated_output['overall_score']:.2f}")
            for provider in evaluated_output["feedback"]:
                print(f"  Feedback from {provider['ai']} ({provider['model']}):")
                for rubric_feedback in provider["feedback"]:
                    print(f"    {rubric_feedback['rubric']}: {rubric_feedback['value']}/5")
                    print(f"      {rubric_feedback['feedback']}")
        print("===============================================")

    output_path = Path(__file__).with_name("votingLayerOutput.json")
    print(f"Saved last run to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
