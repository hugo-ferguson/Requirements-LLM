from __future__ import annotations

from typing import TypeVar

from pydantic_ai import Agent

from app.config import Settings
from app.llm_config import get_models_config
from llm.spec import build_model, with_output_mode

OutputT = TypeVar("OutputT")


class GenerationError(RuntimeError):
    """
    Raised when a PydanticAI agent could not be reached or returned nothing
    usable. Shared by every agent-backed service (AC generation, chat, AC/UAT
    regeneration, UAT generation) so routes have exactly one exception type
    to map to HTTP 502.
    """


def build_agent(
    settings: Settings, output_type: type[OutputT], system_prompt: str
) -> Agent[None, OutputT]:
    """Builds a single-output-type PydanticAI agent against `chat_model`."""
    chat_model = get_models_config().chat_model
    return Agent(
        build_model(chat_model),
        output_type=with_output_mode(output_type, chat_model.output_mode),
        system_prompt=system_prompt,
        model_settings=chat_model.model_settings(),
    )
