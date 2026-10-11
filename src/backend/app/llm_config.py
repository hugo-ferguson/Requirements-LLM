"""Which models the app uses, and for what (backlog R8).

Every model choice lives in one JSON file, config/models.json, so changing a
model is a config edit, not a code change:

- `chat_model`: the chat assistant and single-item regeneration.
- `vision_model`: reads text out of uploaded images.
- `generation_agents`: the ensemble that writes criteria and UAT cases.
- `judges`: the voting layer's scorers (see `voting.models.JudgeConfig`).

Every one of them is a `llm.spec.ModelSpec`, e.g.
`{"provider": "anthropic", "model": "claude-sonnet-5-5"}`, built by
`llm.spec.build_model`.

API keys are never in that file — only the *name* of the environment variable
holding a key, which is read at build time. .env keeps the keys,
infrastructure and tuning numbers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm.spec import ModelSpec
from voting.models import JudgeConfig


class GenerationAgentConfig(ModelSpec):
    """One entry in `generation_agents`."""

    id: str = Field(
        description="Stable label used in logs and results, e.g. 'gemini-flash'")
    enabled: bool = True


class ModelsConfig(BaseModel):
    # A misspelt or out-of-date key fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")

    chat_model: ModelSpec
    vision_model: ModelSpec
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
        raise ModelsConfigError(
            f"Model config not found at {config_path.resolve()}. "
            "Copy config/models.example.json to that path."
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
