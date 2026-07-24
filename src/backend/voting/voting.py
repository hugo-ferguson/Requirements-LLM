from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
RUBRIC_DIR = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer"


class EvaluationInput(BaseModel):
	ai: str
	model: str
	prompt: str
	output: list[str] = Field(min_length=1)
	reference_answer: str = ""


class PrometheusVote(BaseModel):
	feedback: str
	score: int = Field(ge=1, le=5)


class Vote(PrometheusVote):
	output_index: int
	output: str
	rubric: str


class VotingResult(BaseModel):
	ai: str
	model: str
	votes: list[Vote]


class OllamaPrometheusClient:
	def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 120.0) -> None:
		# Set OLLAMA_BASE_URL and PROMETHEUS_MODEL in .env or your shell.
		self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
		self.model = model or os.getenv("PROMETHEUS_MODEL", "ggozad/prometheus2:latest")
		self.timeout = timeout

	async def evaluate(
		self,
		*,
		instruction: str,
		response: str,
		reference_answer: str,
		rubric_name: str,
	) -> Vote:
		rubric = (RUBRIC_DIR / f"{rubric_name}.txt").read_text(encoding="utf-8")
		prompt = rubric.replace("{orig_instruction}", instruction)
		prompt = prompt.replace("{orig_response}", response)
		prompt = prompt.replace("{orig_reference_answer}", reference_answer)
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


async def evaluate_input(evaluation_input: EvaluationInput, client: OllamaPrometheusClient | None = None) -> VotingResult:
	evaluator = client or OllamaPrometheusClient()

	async def evaluate_output(output_index: int, output: str) -> list[Vote]:
		votes = await asyncio.gather(
			*(evaluator.evaluate(
				instruction=evaluation_input.prompt,
				response=output,
				reference_answer=evaluation_input.reference_answer,
				rubric_name=rubric_name,
			) for rubric_name in RUBRIC_NAMES)
		)
		return [
			Vote(
				output_index=output_index,
				output=output,
				rubric=rubric_name,
				feedback=vote.feedback,
				score=vote.score,
			)
			for rubric_name, vote in zip(RUBRIC_NAMES, votes, strict=True)
		]

	output_votes = await asyncio.gather(
		*(evaluate_output(index, output) for index, output in enumerate(evaluation_input.output))
	)
	return VotingResult(
		ai=evaluation_input.ai,
		model=evaluation_input.model,
		votes=[vote for votes in output_votes for vote in votes],
	)


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
