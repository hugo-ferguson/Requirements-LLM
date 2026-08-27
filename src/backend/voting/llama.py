from __future__ import annotations

from voting.combined_provider import OllamaCombinedClient, evaluate_with_combined_model
from voting.models import EvaluationInput, VotingResult


async def evaluate_with_llama(evaluation_input: EvaluationInput) -> VotingResult:
    client = OllamaCombinedClient(
        provider_name="Llama",
        default_model="llama3.1:8b",
        model_env="LLAMA_MODEL",
    )
    return await evaluate_with_combined_model(evaluation_input, client)
