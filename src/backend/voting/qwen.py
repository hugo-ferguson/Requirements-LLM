from __future__ import annotations

from voting.combined_provider import OllamaCombinedClient, evaluate_with_combined_model
from voting.models import EvaluationInput, VotingResult


async def evaluate_with_qwen(evaluation_input: EvaluationInput) -> VotingResult:
    client = OllamaCombinedClient(
        provider_name="Qwen",
        default_model="qwen2.5:7b",
        model_env="QWEN_MODEL",
    )
    return await evaluate_with_combined_model(evaluation_input, client)
