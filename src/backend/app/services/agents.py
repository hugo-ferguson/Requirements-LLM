"""Agent construction and the shared agent-failure error."""

from __future__ import annotations

from pydantic_ai import Agent

from app.llm_config import get_models_config
from llm.spec import build_model, with_output_mode


class GenerationError(RuntimeError):
    """Raised when an agent can't be reached or returns nothing usable.

    Routes map it to HTTP 502. Chat never raises it.
    """


def build_agent[OutputT](
    output_type: type[OutputT], system_prompt: str
) -> Agent[None, OutputT]:
    """Build a PydanticAI agent on the configured `chat_model`.

    Raises:
        ModelsConfigError: config/models.json is missing or invalid.
        ModelBuildError: The provider is unknown or its API key is missing.
    """
    chat_model = get_models_config().chat_model
    return Agent(
        build_model(chat_model),
        output_type=with_output_mode(output_type, chat_model.output_mode),
        system_prompt=system_prompt,
        model_settings=chat_model.model_settings(),
    )
