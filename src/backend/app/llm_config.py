"""Load config/models.json, which sets the models the app uses (R8).

- `chat_model`: chat and single-item regeneration.
- `vision_model`: reads text from uploaded images.
- `generation_agents`: write criteria and UAT cases.
- `judges`: score candidates in the voting layer.

API keys stay in .env. The file only names the variable holding each key.
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
        description="Stable label used in logs and results, e.g. 'gemini-flash'"
    )
    enabled: bool = True


class ModelsConfig(BaseModel):
    """The parsed contents of config/models.json."""

    # Reject unknown keys.
    model_config = ConfigDict(extra="forbid")

    chat_model: ModelSpec
    vision_model: ModelSpec
    generation_agents: list[GenerationAgentConfig]
    judges: list[JudgeConfig]

    def enabled_agents(self) -> list[GenerationAgentConfig]:
        """Return the generation agents not switched off in config."""
        return [agent for agent in self.generation_agents if agent.enabled]

    def enabled_judges(self) -> list[JudgeConfig]:
        """Return the voting judges not switched off in config."""
        return [judge for judge in self.judges if judge.enabled]


class ModelsConfigError(RuntimeError):
    """Raised when the models config is missing, invalid or unusable."""


def load_models_config(path: str | Path) -> ModelsConfig:
    """Read and validate the models config file.

    Raises:
        ModelsConfigError: The file is missing or invalid, repeats an ID, or
            enables no agents or judges.
    """
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
            f"Model config {config_path} is not valid JSON: {error}"
        ) from error

    try:
        config = ModelsConfig.model_validate(payload)
    except ValidationError as error:
        raise ModelsConfigError(
            f"Model config {config_path} is invalid: {error}"
        ) from error

    for section, entry_ids in (
        ("agent", [agent.id for agent in config.generation_agents]),
        ("judge", [judge.id for judge in config.judges]),
    ):
        duplicates = {
            entry_id for entry_id in entry_ids if entry_ids.count(entry_id) > 1
        }
        if duplicates:
            raise ModelsConfigError(
                f"Duplicate {section} ids in {config_path}: {sorted(duplicates)}"
            )

    if not config.enabled_agents():
        raise ModelsConfigError(f"No enabled generation_agents in {config_path}.")
    if not config.enabled_judges():
        raise ModelsConfigError(f"No enabled judges in {config_path}.")

    return config


@lru_cache(maxsize=1)
def get_models_config() -> ModelsConfig:
    """Return the process-wide models config, read once on first use."""
    from app.config import settings

    return load_models_config(settings.models_config_path)
