"""Example caller for the local Prometheus voting evaluator.

Run from src/backend with:

    uv run python -m voting.exampleCall


Each input can select the providers to use:

"providers": ["prometheus", "qwen"]

Valid values are `prometheus`, `qwen`, and `llama`. If `providers` is omitted,
all three are used. Prometheus makes four calls per output, while Qwen and
Llama make one combined call per output. Each result includes averages for all
four rubrics across the selected providers in `rubric_averages`, plus the
overall average in `overall_score`.
	
"""

import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
	sys.path.insert(0, str(BACKEND_DIR))

from voting.voting import EvaluationInput, evaluate_inputs


async def main() -> None:

	#also works with just 1 input
	# input_path = Path(__file__).with_name("exampleInput.json")
	input_path = Path(__file__).with_name("exampleBatchInput.json")
	payload = json.loads(input_path.read_text(encoding="utf-8"))

	# Select only two of the three available providers for this evaluation as an example
	# Valid values are: "prometheus", "qwen", and "llama", "gemini", "claude".
	if isinstance(payload, list):
		for item in payload:
			item["providers"] = ["gemini"]
	else:
		payload["providers"] = ["gemini"]

	# if isinstance(payload, list):
	# 	for item in payload:
	# 		item["providers"] = ["claude"]
	# else:
	# 	payload["providers"] = ["claude"]

	inputs = payload if isinstance(payload, list) else [payload]
	results = await evaluate_inputs([EvaluationInput.model_validate(item) for item in inputs])

	# example with feedback text
	for result in results:
		print(f"AI: {result.ai} | Model: {result.model}")
		for evaluated_output in result.output:
			print(f"Output: {evaluated_output.output}")
			print(f"Overall score: {evaluated_output.overall_score:.2f}")
			for provider in evaluated_output.feedback:
				print(f"  Feedback from {provider.ai} ({provider.model}):")
				for rubric_feedback in provider.feedback:
					print(f"    {rubric_feedback.rubric}: {rubric_feedback.value}/5")
					print(f"      {rubric_feedback.feedback}")

	# #example with no feedback text
	# for result in results:
	# 	print(f"AI: {result.ai} | Model: {result.model}")
	# 	for evaluated_output in result.output:
	# 		print(f"Output: {evaluated_output.output}")
	# 		print("Rubric averages:")
	# 		for rubric_average in evaluated_output.rubric_averages:
	# 			print(f"  {rubric_average.rubric}: {rubric_average.value:.2f}/5")
	# 		print(f"Overall average score: {evaluated_output.overall_score:.2f}")
	# 		for provider in evaluated_output.feedback:
	# 			print(f"  Feedback from {provider.ai} ({provider.model}):")
	# 			for rubric_feedback in provider.feedback:
	# 				print(f"    {rubric_feedback.rubric}: {rubric_feedback.value}/5")
	# 		print() 
	# 	print("===============================================")

	#save the results to a json file to demonstrate the output structure
	output_path = Path(__file__).with_name("exampleOutput.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump([result.model_dump() for result in results], f, indent=2)
	





if __name__ == "__main__":
	asyncio.run(main())