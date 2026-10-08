"""One model entry in config/models.json, and the one place it becomes a model.

Every model the app uses (chat, vision, each generation agent, each judge) is
declared the same way, `{"provider": ..., "model": ...}` plus optional
settings, and `build_model` turns any of them into a PydanticAI model without
knowing which provider it is. PydanticAI routes by provider name, and its
per-model profiles drop settings a model rejects. Where a profile is wrong or
out of date, the fix is a field on the entry (`temperature`, `output_mode`),
not a branch here.
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

# How a model is asked for structured output:
# - "native": the provider's JSON-schema mode (Anthropic's `output_config`,
#   OpenAI's `text.format`, Gemini's `response_schema`, Ollama's `format`).
#   Sends no tools, so it also works for Claude models that reject a forced
#   tool call, and for small local models that write the tool-call envelope
#   as plain text instead of calling it.
# - "tool": a forced call to an output tool, for models without a JSON-schema
#   mode.
# - "prompted": the schema goes in the prompt and the reply is parsed, for
#   models with neither.
OutputMode = Literal["native", "tool", "prompted"]


def with_output_mode(output_type: Any, mode: OutputMode) -> Any:
	"""Wrap an agent's `output_type` for `mode`. Plain-text output is left alone."""
	if output_type is str or mode == "tool":
		return output_type
	return NativeOutput(output_type) if mode == "native" else PromptedOutput(output_type)


class ModelBuildError(RuntimeError):
	"""Raised when a model entry can't be built: unknown provider or missing key."""


class ModelSpec(BaseModel):
	"""Which model to call, and how. The shared shape of every model entry."""

	# A misspelt or unknown key fails loudly instead of being ignored.
	model_config = ConfigDict(extra="forbid")

	provider: str = Field(
		description="A PydanticAI provider name, e.g. 'anthropic', 'openai', 'google', "
		"'ollama'; 'test' for an offline stub"
	)
	model: str = Field(description="The provider's model name, e.g. 'claude-sonnet-5-5'")
	# The environment variable holding the key. None leaves it to the
	# provider's standard variable: ANTHROPIC_API_KEY, OPENAI_API_KEY,
	# GEMINI_API_KEY, OLLAMA_BASE_URL and so on.
	api_key_env: str | None = None
	base_url: str | None = None
	# None sends no temperature at all. Claude from Opus 4.7 / Sonnet 5 on and
	# OpenAI reasoning models reject any non-default value.
	temperature: float | None = Field(default=None, ge=0.0, le=2.0)
	output_mode: OutputMode = "native"

	@property
	def name(self) -> str:
		"""`provider:model`, for logs and error messages."""
		return f"{self.provider}:{self.model}"

	def api_key(self) -> str | None:
		"""Resolve the key from the environment at call time, not import time."""
		if self.api_key_env is None:
			return None
		return os.environ.get(self.api_key_env)

	def model_settings(self, **extra: Any) -> ModelSettings:
		"""This entry's sampling settings, plus any the caller adds.

		Provider-prefixed settings such as `anthropic_cache_instructions` are
		safe to pass for every provider: the others ignore them.
		"""
		settings = dict(extra)
		if self.temperature is not None:
			settings["temperature"] = self.temperature
		return cast(ModelSettings, settings)


def build_model(spec: ModelSpec, *, http_client: httpx2.AsyncClient | None = None) -> Model:
	"""The PydanticAI model for `spec`, whichever provider it names.

	`http_client` replaces the provider's own HTTP client, e.g. to put a mock
	transport under a test.
	"""
	if spec.provider == "test":
		from pydantic_ai.models.test import TestModel

		return TestModel()

	kwargs: dict[str, Any] = {}
	if spec.api_key_env is not None:
		api_key = spec.api_key()
		if not api_key:
			raise ModelBuildError(f"{spec.name} needs {spec.api_key_env} set in the environment.")
		kwargs["api_key"] = api_key
	if spec.base_url:
		kwargs["base_url"] = spec.base_url
	if http_client is not None:
		kwargs["http_client"] = http_client

	try:
		return infer_model(spec.name, provider_factory=lambda name: infer_provider_class(name)(**kwargs))
	except (UserError, ImportError, ValueError) as error:
		# Unknown provider, a provider SDK that isn't installed, or the
		# provider's own check for its standard key variable.
		raise ModelBuildError(f"Cannot build {spec.name}: {error}") from error
