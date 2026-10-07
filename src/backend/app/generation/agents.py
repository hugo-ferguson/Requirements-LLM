"""Builds one PydanticAI agent per roster entry.

Every agent shares the same `deps_type` and system prompt, and every agent in a
given pass shares the same `output_type`. Only the underlying model changes.
That shared contract is what makes the ensemble outputs comparable, and it is
why adding a model is a config edit rather than a code change.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from app.llm_config import GenerationAgentConfig, with_output_mode
from app.generation.models import (
    GeneratedCriteriaSet,
    GeneratedCriterion,
    GeneratedUatCase,
    GeneratedUatCaseSet,
    NumberedCriteriaSet,
    NumberedCriterion,
    NumberedUatCase,
    NumberedUatCaseSet,
)
from app.generation.prompts import SYSTEM_PROMPT, GenerationDeps

GenerationAgent = Agent[GenerationDeps, GeneratedCriteriaSet]
GenerationOutput = (
    type[GeneratedCriteriaSet]
    | type[NumberedCriteriaSet]
    | type[GeneratedUatCaseSet]
    | type[NumberedUatCaseSet]
)


class AgentBuildError(RuntimeError):
    """Raised when an agent cannot be constructed (missing key, missing extra)."""


def _stub_output(config: GenerationAgentConfig, output_type: GenerationOutput = GeneratedCriteriaSet):
    """Deterministic output for the `test` provider.

    Two criteria are identical across every test agent and one is seeded with
    the agent id, so a two-agent test roster yields three distinct pooled
    candidates and still exercises the cross-agent dedupe path. As a pass-2
    follower it answers titles 1-3 with the same text, by number. UAT output
    follows the same pattern with two cases.
    """
    if output_type in (GeneratedUatCaseSet, NumberedUatCaseSet):
        return _stub_uat_output(config, output_type)
    full = _stub_criteria(config)
    if output_type is NumberedCriteriaSet:
        return NumberedCriteriaSet(
            criteria=[
                NumberedCriterion(title_number=n, given=c.given, when=c.when, then=c.then)
                for n, c in enumerate(full.criteria, start=1)
            ]
        )
    return full


def _stub_uat_output(
    config: GenerationAgentConfig, output_type: GenerationOutput
) -> GeneratedUatCaseSet | NumberedUatCaseSet:
    cases = [
        GeneratedUatCase(
            title="Log in with valid credentials",
            description=f"{config.id}: enter a registered email and password; the home page opens",
        ),
        GeneratedUatCase(
            title="Reject a wrong password",
            description=f"{config.id}: enter a wrong password; an error shows and login fails",
        ),
    ]
    if output_type is NumberedUatCaseSet:
        return NumberedUatCaseSet(
            cases=[
                NumberedUatCase(title_number=n, description=c.description)
                for n, c in enumerate(cases, start=1)
            ]
        )
    return GeneratedUatCaseSet(cases=cases)


def _stub_criteria(config: GenerationAgentConfig) -> GeneratedCriteriaSet:
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


def _build_model(
    config: GenerationAgentConfig, output_type: GenerationOutput = GeneratedCriteriaSet
) -> Model:
    """Translate a roster entry into a concrete PydanticAI model object.

    Provider SDKs are imported inside each branch so that a provider the team
    isn't using doesn't need to be installed.
    """
    if config.provider == "test":
        from pydantic_ai.models.test import TestModel

        return TestModel(custom_output_args=_stub_output(config, output_type))

    if config.provider in ("openai", "ollama"):
        try:
            from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
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
            return OpenAIChatModel(
                config.model,
                provider=OpenAIProvider(base_url=base_url, api_key="ollama"),
            )

        api_key = config.api_key()
        if not api_key:
            raise AgentBuildError(
                f"Agent {config.id!r} needs {config.api_key_env} set in the environment."
            )
        # OpenAI's reasoning models reject function tools on /v1/chat/completions
        # unless reasoning is switched off, and structured output here is a tool
        # call. The Responses API accepts both, so real OpenAI goes there.
        return OpenAIResponsesModel(
            config.model,
            provider=OpenAIProvider(api_key=api_key, base_url=config.base_url),
        )

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


def build_agent(
    config: GenerationAgentConfig,
    output_type: GenerationOutput = GeneratedCriteriaSet,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    with_context: bool = True,
) -> GenerationAgent:
    """Construct a generation agent for one roster entry.

    `output_type` is `NumberedCriteriaSet` for pass-2 followers in
    title-anchored generation, and the full criteria set everywhere else.
    UAT generation passes its own output types and system prompt, and
    `with_context=False`: it works from one AC, not the user story and its
    retrieved documents.
    """
    agent: GenerationAgent = Agent(
        _build_model(config, output_type),
        output_type=with_output_mode(output_type, config.output_mode),
        deps_type=GenerationDeps,
        system_prompt=system_prompt,
        model_settings=(
            ModelSettings() if config.temperature is None
            else ModelSettings(temperature=config.temperature)
        ),
        retries=2,
    )

    if not with_context:
        return agent

    @agent.system_prompt
    def inject_project_context(ctx: RunContext[GenerationDeps]) -> str:
        """Appended to the base system prompt at call time (backlog R11)."""
        return ctx.deps.context_block()

    @agent.system_prompt
    def inject_reviewer_feedback(ctx: RunContext[GenerationDeps]) -> str:
        """Appended only on a regeneration pass (backlog R20)."""
        return ctx.deps.feedback_block() or ""

    return agent
