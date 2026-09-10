from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import litellm

from voting.models import CombinedVote, EvaluationInput, EvaluatedOutput, ProviderFeedback, RubricAverage, RubricFeedback, VotingResult


RUBRIC_NAMES = ("correctness", "coverage", "relevance", "understandability")
COMBINED_PROMPT = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer" / "combined.txt"

COMBINED_VOTE_SCHEMA = {
	"type": "json_schema",
	"json_schema": {
		"name": "CombinedVote",
		"schema": CombinedVote.model_json_schema(),
	},
}


def _strip_markdown_fences(text: str) -> str:
	"""Remove markdown code fences that some models wrap JSON output in."""
	stripped = text.strip()
	if stripped.startswith("```"):
		stripped = stripped.strip("`")
		if stripped.lower().startswith("json"):
			stripped = stripped[4:].lstrip()
	match = re.search(r"\{.*\}", stripped, re.DOTALL)
	if match:
		return match.group(0)
	return stripped


class LiteLLMCombinedClient:
	"""Evaluates acceptance criteria using any LiteLLM-supported model."""

	def __init__(
		self,
		provider_name: str,
		model: str,
		*,
		temperature: float = 0.1,
		timeout: float = 120.0,
		num_retries: int = 2,
	) -> None:
		self.provider_name = provider_name
		self.model = model
		self.temperature = temperature
		self.timeout = timeout
		self.num_retries = num_retries

	async def evaluate(self, *, instruction: str, response: str) -> CombinedVote:
		prompt = COMBINED_PROMPT.read_text(encoding="utf-8")
		prompt = prompt.replace("{orig_instruction}", instruction)
		prompt = prompt.replace("{orig_response}", response)

		schema_hint = json.dumps(CombinedVote.model_json_schema(), indent=2)

		result = await litellm.acompletion(
			model=self.model,
			messages=[
				{
					"role": "system",
					"content": (
						"You are an acceptance-criteria evaluator. "
						"Return only valid JSON matching this schema:\n"
						f"{schema_hint}"
					),
				},
				{"role": "user", "content": prompt},
			],
			response_format=COMBINED_VOTE_SCHEMA,
			temperature=self.temperature,
			timeout=self.timeout,
			num_retries=self.num_retries,
		)

		content = result.choices[0].message.content
		if not isinstance(content, str) or not content.strip():
			raise ValueError(f"{self.model} returned no message content")

		cleaned = _strip_markdown_fences(content)
		return CombinedVote.model_validate_json(cleaned)


def _error_output(output: str, provider_name: str, model: str, message: str) -> EvaluatedOutput:
	feedback = [RubricFeedback(rubric=name, value=-1, feedback=message) for name in RUBRIC_NAMES]
	provider_feedback = ProviderFeedback(
		ai=provider_name,
		model=model,
		feedback=feedback,
		overall_score=-1.0,
	)
	return EvaluatedOutput(
		output=output,
		feedback=[provider_feedback],
		rubric_averages=[RubricAverage(rubric=name, value=-1.0) for name in RUBRIC_NAMES],
		overall_score=-1.0,
	)


async def evaluate_with_combined_model(evaluation_input: EvaluationInput, client: LiteLLMCombinedClient) -> VotingResult:
	async def evaluate_output(output: str) -> EvaluatedOutput:
		result = await client.evaluate(instruction=evaluation_input.prompt, response=output)
		rubric_feedback = [
			RubricFeedback(
				rubric=name,
				value=getattr(result, name).score,
				feedback=getattr(result, name).feedback,
			)
			for name in RUBRIC_NAMES
		]
		overall_score = sum(item.value for item in rubric_feedback) / len(rubric_feedback)
		provider_feedback = ProviderFeedback(
			ai=client.provider_name,
			model=client.model,
			feedback=rubric_feedback,
			overall_score=overall_score,
		)
		return EvaluatedOutput(
			output=output,
			feedback=[provider_feedback],
			rubric_averages=[RubricAverage(rubric=item.rubric, value=float(item.value)) for item in rubric_feedback],
			overall_score=overall_score,
		)

	results = await asyncio.gather(
		*(evaluate_output(output) for output in evaluation_input.output),
		return_exceptions=True,
	)
	evaluated_outputs: list[EvaluatedOutput] = []
	for output, result in zip(evaluation_input.output, results, strict=True):
		if isinstance(result, Exception):
			evaluated_outputs.append(
				_error_output(
					output,
					client.provider_name,
					client.model,
					f"Error evaluating output: {result.__class__.__name__}: {result}",
				)
			)
		else:
			evaluated_outputs.append(result)
	return VotingResult(
		ai=evaluation_input.ai,
		model=evaluation_input.model,
		prompt=evaluation_input.prompt,
		output=evaluated_outputs,
	)
