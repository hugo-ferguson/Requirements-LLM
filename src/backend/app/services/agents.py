from __future__ import annotations

from typing import TypeVar

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from app.config import Settings
from app.llm_config import get_models_config, with_output_mode

OutputT = TypeVar("OutputT")


class GenerationError(RuntimeError):
    """
    Raised when a PydanticAI agent could not be reached or returned nothing
    usable. Shared by every agent-backed service (AC generation, chat, AC/UAT
    regeneration, UAT generation) so routes have exactly one exception type
    to map to HTTP 502.
    """


def resolve_model(settings: Settings) -> Model | str:
    """
    Resolves `chat_model` from config/models.json into the concrete
    PydanticAI model object.

    The API key is passed explicitly rather than left to the ambient
    environment, so the app has one place that decides where credentials
    come from. Centralised here so every agent-backed service shares this
    logic instead of repeating GoogleModel/GoogleProvider construction.
    """
    chat_model = get_models_config().chat_model
    provider, _, model_name = chat_model.partition(":")

    if provider == "google":
        return GoogleModel(
            model_name,  # type: ignore[arg-type]
            provider=GoogleProvider(api_key=settings.gemini_api_key),
        )

    # Any other PydanticAI model string still works, but it has to find its
    # own credentials in the environment.
    return chat_model  # type: ignore[return-value]


def build_agent(
    settings: Settings, output_type: type[OutputT], system_prompt: str
) -> Agent[None, OutputT]:
    """Builds a single-output-type PydanticAI agent against the configured model."""
    output_mode = get_models_config().chat_output_mode
    return Agent(
        resolve_model(settings),
        output_type=with_output_mode(output_type, output_mode),
        system_prompt=system_prompt,
    )
