from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from voting.models import EvaluationInput, Vote, VotingResult
from voting.prometheus import evaluate_with_prometheus


async def evaluate_input(evaluation_input: EvaluationInput) -> VotingResult:
	return await evaluate_with_prometheus(evaluation_input)


mcp = FastMCP("requirements-voting")


@mcp.tool()
async def evaluate_acceptance_criteria(evaluation_input: EvaluationInput) -> VotingResult:
	"""Score every generated acceptance criterion against all four voting rubrics."""
	return await evaluate_input(evaluation_input)


async def main(input_path: str) -> None:
	payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
	result = await evaluate_input(EvaluationInput.model_validate(payload))
	print(result.model_dump_json(indent=2))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate generated criteria with Prometheus through Ollama")
	parser.add_argument("input", nargs="?", default=str(Path(__file__).with_name("exampleInput.json")))
	parser.add_argument("--mcp", action="store_true", help="Run the MCP server over stdio")
	args = parser.parse_args()
	if args.mcp:
		mcp.run()
	else:
		asyncio.run(main(args.input))
