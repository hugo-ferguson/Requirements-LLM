"""Agent roster configuration for the ensemble (backlog R8).

Agents are declared in a JSON file, not in code, so the pool can be changed
without a code edit. API keys are never in that file — only the *name* of the
environment variable holding the key, which is read at build time.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["openai", "anthropic", "google", "ollama", "test"]


class GenerationAgentConfig(BaseModel):
    """One entry in the roster."""

    id: str = Field(
        description="Stable label used in logs and results, e.g. 'gemini-flash'")
    provider: ProviderName
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    enabled: bool = True

    def api_key(self) -> str | None:
        """Resolve the key from the environment at call time, not import time."""
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env)


class GenerationRoster(BaseModel):
    agents: list[GenerationAgentConfig] = Field(default_factory=list)

    def enabled_agents(self) -> list[GenerationAgentConfig]:
        return [agent for agent in self.agents if agent.enabled]


class RosterConfigError(RuntimeError):
    """Raised when the roster file is missing, malformed, or empty."""


def load_roster(path: str | Path) -> GenerationRoster:
    """Read and validate the roster file."""
    roster_path = Path(path)
    if not roster_path.is_file():
        raise RosterConfigError(
            f"Generation roster file not found at {roster_path.resolve()}. "
            "Copy config/generation_agents.example.json to that path."
        )

    try:
        payload = json.loads(roster_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RosterConfigError(
            f"Roster file {roster_path} is not valid JSON: {error}") from error

    roster = GenerationRoster.model_validate(payload)

    ids = [agent.id for agent in roster.agents]
    duplicates = {name for name in ids if ids.count(name) > 1}
    if duplicates:
        raise RosterConfigError(
            f"Duplicate agent ids in roster: {sorted(duplicates)}")

    if not roster.enabled_agents():
        raise RosterConfigError(f"No enabled agents in {roster_path}.")

    return roster


@lru_cache(maxsize=1)
def get_roster() -> GenerationRoster:
    """Process-wide roster, read once on first use."""
    from app.config import settings

    return load_roster(settings.generation_roster_path)
