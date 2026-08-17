from __future__ import annotations

import argparse
import asyncio
import json
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
from voting.claude import evaluate_with_claude
from voting.gemini import evaluate_with_gemini
from voting.llama import evaluate_with_llama
from voting.prometheus import evaluate_with_prometheus
from voting.qwen import evaluate_with_qwen


RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")


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
			rubric_averages=[RubricAverage(rubric=rubric, value=-1.0) for rubric in RUBRIC_NAMES],
			overall_score=-1.0,
		)
		for output in evaluation_input.output
	]


def _merge_provider_outputs(evaluation_input: EvaluationInput, provider_outputs: list[list[EvaluatedOutput]]) -> VotingResult:
	evaluated_outputs: list[EvaluatedOutput] = []
	for output_index, output in enumerate(evaluation_input.output):
		feedback = [outputs[output_index].feedback[0] for outputs in provider_outputs]
		valid_scores = [provider.overall_score for provider in feedback if provider.overall_score >= 0]
		overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else -1.0
		rubric_averages = []
		for rubric in RUBRIC_NAMES:
			values = [
				entry.value
				for provider in feedback
				for entry in provider.feedback
				if entry.rubric == rubric and entry.value >= 0
			]
			rubric_averages.append(
				RubricAverage(rubric=rubric, value=sum(values) / len(values) if values else -1.0)
			)
		evaluated_outputs.append(
			EvaluatedOutput(
				output=output,
				feedback=feedback,
				rubric_averages=rubric_averages,
				overall_score=overall_score,
			)
		)
	return VotingResult(
		ai=evaluation_input.ai,
		model=evaluation_input.model,
		prompt=evaluation_input.prompt,
		output=evaluated_outputs,
	)


async def evaluate_input(evaluation_input: EvaluationInput) -> VotingResult:
	provider_details = {
		"prometheus": ("Prometheus", "ggozad/prometheus2:latest", evaluate_with_prometheus),
		"qwen": ("Qwen", "qwen2.5:7b", evaluate_with_qwen),
		"llama": ("Llama", "llama3.1:8b", evaluate_with_llama),
		"gemini": ("Gemini", "gemini-flash-latest", evaluate_with_gemini),
		"claude": ("Claude", "claude-3-5-sonnet-20241022", evaluate_with_claude),
	}
	selected = [provider_details[name] for name in evaluation_input.providers]
	provider_results = await asyncio.gather(
		*(evaluate(evaluation_input) for _, _, evaluate in selected),
		return_exceptions=True,
	)

	provider_outputs: list[list[EvaluatedOutput]] = []
	for result, (provider, model, _) in zip(provider_results, selected, strict=True):
		if isinstance(result, Exception):
			provider_outputs.append(_provider_error_outputs(evaluation_input, provider, model, result))
		elif isinstance(result, VotingResult):
			provider_outputs.append(result.output)
		else:
			provider_outputs.append(result)

	return _merge_provider_outputs(evaluation_input, provider_outputs)


async def evaluate_inputs(evaluation_inputs: list[EvaluationInput]) -> list[VotingResult]:
	"""Evaluate every input independently and preserve input order."""
	return await asyncio.gather(*(evaluate_input(item) for item in evaluation_inputs))


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
	parser = argparse.ArgumentParser(description="Evaluate generated criteria with Prometheus through Ollama")
	parser.add_argument("input", nargs="?", default=str(Path(__file__).with_name("exampleInput.json")))
	parser.add_argument("--mcp", action="store_true", help="Run the MCP server over stdio")
	args = parser.parse_args()
	if args.mcp:
		mcp.run()
	else:
		asyncio.run(main(args.input))
