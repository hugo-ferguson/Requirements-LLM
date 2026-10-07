"""Which models the app uses, and for what (backlog R8).

Every model choice lives in one JSON file, config/models.json, so changing a
model is a config edit, not a code change:

- `chat_model`: the chat assistant and single-item regeneration
  (a PydanticAI model string, e.g. "anthropic:claude-sonnet-5-5").
- `vision_model`: reads text out of uploaded images
  (a LiteLLM model string, e.g. "anthropic/claude-sonnet-5-5").
- `generation_agents`: the ensemble that writes criteria and UAT cases.
- `judges`: the voting layer's scorers (see `voting.models.JudgeConfig`).

API keys are never in that file — only the *name* of the environment variable
holding a key, which is read at build time. .env keeps the keys,
infrastructure and tuning numbers.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import NativeOutput, PromptedOutput

from voting.models import JudgeConfig

logger = logging.getLogger(__name__)

ProviderName = Literal["openai", "anthropic", "google", "ollama", "test"]

# How a model is asked for structured output:
# - "tool": PydanticAI's default, a forced call to an output tool. Claude from
#   Sonnet 5.5 / Opus 5.5 on rejects forced tool calls with a 400.
# - "native": the provider's JSON-schema mode (Anthropic's `output_config`,
#   Ollama's `format`). Also the reliable choice for small local models, which
#   tend to write the tool-call envelope as plain text instead of calling it.
# - "prompted": the schema goes in the prompt and the reply is parsed, for
#   models with neither.
OutputMode = Literal["tool", "native", "prompted"]


def with_output_mode(output_type: Any, mode: OutputMode) -> Any:
    """Wrap an agent's `output_type` for `mode`. Plain-text output is left alone."""
    if output_type is str or mode == "tool":
        return output_type
    return NativeOutput(output_type) if mode == "native" else PromptedOutput(output_type)

# Settings that used to live in .env and are now in config/models.json. Still
# being set means someone's .env predates the move, and their value is ignored.
RETIRED_ENV_VARS = {
    "LLM_MODEL": "chat_model",
    "VISION_MODEL": "vision_model",
    "VOTING_PROVIDERS": "judges",
    "VOTING_JUDGES": "judges (set `enabled`)",
    "PROMETHEUS_MODEL": 'judges (a judge with "style": "per_rubric")',
    "GENERATION_ROSTER_PATH": "MODELS_CONFIG_PATH",
}


class GenerationAgentConfig(BaseModel):
    """One entry in `generation_agents`."""

    id: str = Field(
        description="Stable label used in logs and results, e.g. 'gemini-flash'")
    provider: ProviderName
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    # None sends no temperature. Claude from Opus 4.7 / Sonnet 5 on and OpenAI
    # reasoning models reject any non-default value.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    output_mode: OutputMode = "tool"
    enabled: bool = True

    def api_key(self) -> str | None:
        """Resolve the key from the environment at call time, not import time."""
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env)


class ModelsConfig(BaseModel):
    # A misspelt or out-of-date key fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    chat_model: str
    # Output mode for the chat model's structured replies; see `OutputMode`.
    chat_output_mode: OutputMode = "tool"
    vision_model: str
    generation_agents: list[GenerationAgentConfig]
    judges: list[JudgeConfig]

    def enabled_agents(self) -> list[GenerationAgentConfig]:
        return [agent for agent in self.generation_agents if agent.enabled]

    def enabled_judges(self) -> list[JudgeConfig]:
        return [judge for judge in self.judges if judge.enabled]


class ModelsConfigError(RuntimeError):
    """Raised when the models config file is missing, malformed, or unusable."""


def load_models_config(path: str | Path) -> ModelsConfig:
    """Read and validate the models config file."""
    config_path = Path(path)
    if not config_path.is_file():
        hint = ""
        if config_path.with_name("generation_agents.json").is_file():
            hint = (
                " It replaces generation_agents.json, along with LLM_MODEL, "
                "VISION_MODEL, VOTING_PROVIDERS and VOTING_JUDGES in .env."
            )
        raise ModelsConfigError(
            f"Model config not found at {config_path.resolve()}. "
            f"Copy config/models.example.json to that path.{hint}"
        )

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelsConfigError(
            f"Model config {config_path} is not valid JSON: {error}") from error

    try:
        config = ModelsConfig.model_validate(payload)
    except ValidationError as error:
        raise ModelsConfigError(f"Model config {config_path} is invalid: {error}") from error

    for section, ids in (
        ("agent", [agent.id for agent in config.generation_agents]),
        ("judge", [judge.id for judge in config.judges]),
    ):
        duplicates = {name for name in ids if ids.count(name) > 1}
        if duplicates:
            raise ModelsConfigError(
                f"Duplicate {section} ids in {config_path}: {sorted(duplicates)}")

    if not config.enabled_agents():
        raise ModelsConfigError(f"No enabled generation_agents in {config_path}.")
    if not config.enabled_judges():
        raise ModelsConfigError(f"No enabled judges in {config_path}.")

    return config


@lru_cache(maxsize=1)
def get_models_config() -> ModelsConfig:
    """Process-wide models config, read once on first use."""
    from app.config import settings

    return load_models_config(settings.models_config_path)


def warn_about_retired_env_vars() -> None:
    """Say so when .env still sets something that moved to config/models.json."""
    for name, replacement in RETIRED_ENV_VARS.items():
        if name in os.environ:
            logger.warning(
                "%s is set but no longer read; set %s in config/models.json instead",
                name,
                replacement,
            )
