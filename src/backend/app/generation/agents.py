"""Builds one PydanticAI agent per roster entry.

Every agent shares the same `output_type`, `deps_type` and system prompt. Only
the underlying model changes. That shared contract is what makes the ensemble
outputs comparable, and it is why adding a model is a config edit rather than a
code change.
"""

from __future__ import annotations

from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from app.generation.config import GenerationAgentConfig
from app.generation.models import GeneratedCriteriaSet, GeneratedCriterion
from app.generation.prompts import SYSTEM_PROMPT, GenerationDeps

GenerationAgent = Agent[GenerationDeps, GeneratedCriteriaSet]


class AgentBuildError(RuntimeError):
    """Raised when an agent cannot be constructed (missing key, missing extra)."""


def _stub_output(config: GenerationAgentConfig) -> GeneratedCriteriaSet:
    """Deterministic output for the `test` provider.

    Two criteria are identical across every test agent and one is seeded with
    the agent id, so a two-agent test roster yields three distinct pooled
    candidates and still exercises the cross-agent dedupe path.
    """
    return GeneratedCriteriaSet(
        user_story_summary="Stubbed summary for offline tests.",
        criteria=[
            GeneratedCriterion(
                title="Successful login redirects home",
                given="a registered user is on the login page",
                when="they submit valid credentials",
                then="they are redirected to the home page",
            ),
            GeneratedCriterion(
                title="Invalid password shows an error",
                given="a registered user is on the login page",
                when="they submit an incorrect password",
                then="an inline error is shown and they remain on the login page",
            ),
            GeneratedCriterion(
                title=f"Unique candidate from {config.id}",
                given=f"the {config.id} agent produced this candidate",
                when="the ensemble pools every agent's output",
                then="this criterion survives deduplication",
            ),
        ],
    )


def _build_model(config: GenerationAgentConfig) -> Model:
    """Translate a roster entry into a concrete PydanticAI model object.

    Provider SDKs are imported inside each branch so that a provider the team
    isn't using doesn't need to be installed.
    """
    if config.provider == "test":
        from pydantic_ai.models.test import TestModel

        return TestModel(custom_output_args=_stub_output(config))

    if config.provider in ("openai", "ollama"):
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError as error:  # pragma: no cover - depends on extras
            raise AgentBuildError(
                "The OpenAI extra is not installed. Add 'pydantic-ai-slim[openai]'."
            ) from error

        if config.provider == "ollama":
            # Ollama exposes an OpenAI-compatible API. The key is ignored by
            # Ollama but the client requires a non-empty string.
            base_url = (
                config.base_url or "http://localhost:11434/v1").rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            provider = OpenAIProvider(base_url=base_url, api_key="ollama")
        else:
            api_key = config.api_key()
            if not api_key:
                raise AgentBuildError(
                    f"Agent {config.id!r} needs {config.api_key_env} set in the environment."
                )
            provider = OpenAIProvider(
                api_key=api_key, base_url=config.base_url)

        return OpenAIChatModel(config.model, provider=provider)

    if config.provider == "anthropic":
        try:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
        except ImportError as error:  # pragma: no cover - depends on extras
            raise AgentBuildError(
                "The Anthropic extra is not installed. Add 'pydantic-ai-slim[anthropic]'."
            ) from error

        api_key = config.api_key()
        if not api_key:
            raise AgentBuildError(
                f"Agent {config.id!r} needs {config.api_key_env} set in the environment."
            )
        return AnthropicModel(config.model, provider=AnthropicProvider(api_key=api_key))

    if config.provider == "google":
        try:
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider
        except ImportError as error:  # pragma: no cover - depends on extras
            raise AgentBuildError(
                "The Google extra is not installed. Add 'pydantic-ai-slim[google]'."
            ) from error

        api_key = config.api_key()
        if not api_key:
            raise AgentBuildError(
                f"Agent {config.id!r} needs {config.api_key_env} set in the environment."
            )
        return GoogleModel(config.model, provider=GoogleProvider(api_key=api_key))

    raise AgentBuildError(
        f"Unknown provider {config.provider!r} for agent {config.id!r}.")


def _output_type(config: GenerationAgentConfig):
    """Choose how the model is asked to return `GeneratedCriteriaSet`.

    PydanticAI's default is a tool call. Small local models served through
    Ollama are unreliable at that — they emit the tool-call envelope
    (`{"name": "final_result", "arguments": {...}}`) as ordinary text, which
    then fails validation against the schema and burns all the output retries.
    Asking Ollama for a native JSON-schema response instead removes the tool
    round-trip entirely.

    Cloud providers keep the tool-based default, which they handle well and
    which Anthropic requires (it has no native JSON-schema mode).
    """
    if config.provider == "ollama":
        return NativeOutput(GeneratedCriteriaSet)
    return GeneratedCriteriaSet


def build_agent(config: GenerationAgentConfig) -> GenerationAgent:
    """Construct a generation agent for one roster entry."""
    agent: GenerationAgent = Agent(
        _build_model(config),
        output_type=_output_type(config),
        deps_type=GenerationDeps,
        system_prompt=SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=config.temperature),
        retries=2,
    )

    @agent.system_prompt
    def inject_project_context(ctx: RunContext[GenerationDeps]) -> str:
        """Appended to the base system prompt at call time (backlog R11)."""
        return ctx.deps.context_block()

    @agent.system_prompt
    def inject_reviewer_feedback(ctx: RunContext[GenerationDeps]) -> str:
        """Appended only on a regeneration pass (backlog R20)."""
        return ctx.deps.feedback_block() or ""

    return agent
