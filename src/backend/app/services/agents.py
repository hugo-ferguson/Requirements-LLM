from __future__ import annotations

from typing import TypeVar

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from app.config import Settings

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
    Resolves `settings.llm_model` into the concrete PydanticAI model object.

    The API key is passed explicitly rather than left to the ambient
    environment, so the app has one place that decides where credentials
    come from. Centralised here so every agent-backed service shares this
    logic instead of repeating GoogleModel/GoogleProvider construction.
    """
    provider, _, model_name = settings.llm_model.partition(":")

    if provider == "google":
        return GoogleModel(
            model_name,  # type: ignore[arg-type]
            provider=GoogleProvider(api_key=settings.gemini_api_key),
        )

    # Any other PydanticAI model string still works, but it has to find its
    # own credentials in the environment.
    return settings.llm_model  # type: ignore[return-value]


def build_agent(
    settings: Settings, output_type: type[OutputT], system_prompt: str
) -> Agent[None, OutputT]:
    """Builds a single-output-type PydanticAI agent against the configured model."""
    return Agent(resolve_model(settings), output_type=output_type, system_prompt=system_prompt)
