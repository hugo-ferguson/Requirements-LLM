"""Model entries from config/models.json, and building them into models.

Fix provider quirks with a field on the entry, not a branch here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import NativeOutput, PromptedOutput
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    import httpx2

# How to get structured output: the provider's JSON-schema mode ("native"),
# a forced tool call ("tool"), or the schema in the prompt ("prompted").
OutputMode = Literal["native", "tool", "prompted"]


def with_output_mode(output_type: Any, mode: OutputMode) -> Any:
    """Wrap an agent's `output_type` for `mode`. `str` and tool mode stay unwrapped."""
    if output_type is str or mode == "tool":
        return output_type
    return (
        NativeOutput(output_type) if mode == "native" else PromptedOutput(output_type)
    )


class ModelBuildError(RuntimeError):
    """Raised when a model entry can't be built (provider or key)."""


class ModelSpec(BaseModel):
    """One model entry: which model to call, and how."""

    # Reject unknown keys.
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        description="A PydanticAI provider name, e.g. 'anthropic', 'openai', 'google', "
        "'ollama'; 'test' for an offline stub"
    )
    model: str = Field(
        description="The provider's model name, e.g. 'claude-sonnet-5-5'"
    )
    # Env variable holding the key. None uses the provider's default variable.
    api_key_env: str | None = None
    base_url: str | None = None
    # None sends no temperature, as some newer models reject one.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    output_mode: OutputMode = "native"

    @property
    def name(self) -> str:
        """The `provider:model` name PydanticAI resolves."""
        return f"{self.provider}:{self.model}"

    def api_key(self) -> str | None:
        """Read the API key from the environment at call time."""
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env)

    def model_settings(self, **extra: Any) -> ModelSettings:
        """Return this entry's sampling settings plus any extra ones.

        Provider-prefixed settings are safe to pass, as other providers ignore them.
        """
        settings = dict(extra)
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        return cast(ModelSettings, settings)


def build_model(
    spec: ModelSpec, *, http_client: httpx2.AsyncClient | None = None
) -> Model:
    """Build the PydanticAI model for `spec`, whatever its provider.

    Args:
        spec: The model entry to build.
        http_client: Replaces the provider's HTTP client, for tests.

    Raises:
        ModelBuildError: The provider is unknown or not installed, or its
            API key is missing.
    """
    if spec.provider == "test":
        from pydantic_ai.models.test import TestModel

        return TestModel()

    provider_kwargs: dict[str, Any] = {}
    if spec.api_key_env is not None:
        api_key = spec.api_key()
        if not api_key:
            raise ModelBuildError(
                f"{spec.name} needs {spec.api_key_env} set in the environment."
            )
        provider_kwargs["api_key"] = api_key
    if spec.base_url:
        provider_kwargs["base_url"] = spec.base_url
    if http_client is not None:
        provider_kwargs["http_client"] = http_client

    def make_provider(provider_name: str) -> Any:
        return infer_provider_class(provider_name)(**provider_kwargs)

    try:
        return infer_model(spec.name, provider_factory=make_provider)
    except (UserError, ImportError, ValueError) as error:
        # Unknown provider, missing SDK, or missing standard key variable.
        raise ModelBuildError(f"Cannot build {spec.name}: {error}") from error
