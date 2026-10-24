"""Build one PydanticAI agent per roster entry.

Agents in a pass differ only by model, so their outputs are comparable.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

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
from app.llm_config import GenerationAgentConfig
from llm.spec import build_model, with_output_mode

GenerationAgent = Agent[GenerationDeps, GeneratedCriteriaSet]
GenerationOutput = (
    type[GeneratedCriteriaSet]
    | type[NumberedCriteriaSet]
    | type[GeneratedUatCaseSet]
    | type[NumberedUatCaseSet]
)


def _stub_output(
    config: GenerationAgentConfig, output_type: GenerationOutput
) -> (
    GeneratedCriteriaSet
    | NumberedCriteriaSet
    | GeneratedUatCaseSet
    | NumberedUatCaseSet
):
    """Return fixed output for the `test` provider.

    Two criteria are shared by every agent and one is unique, to exercise dedupe.
    """
    if output_type in (GeneratedUatCaseSet, NumberedUatCaseSet):
        return _stub_uat_output(config, output_type)
    criteria_set = _stub_criteria(config)
    if output_type is NumberedCriteriaSet:
        return NumberedCriteriaSet(
            criteria=[
                NumberedCriterion(
                    title_number=number,
                    given=criterion.given,
                    when=criterion.when,
                    then=criterion.then,
                )
                for number, criterion in enumerate(criteria_set.criteria, start=1)
            ]
        )
    return criteria_set


def _stub_uat_output(
    config: GenerationAgentConfig, output_type: GenerationOutput
) -> GeneratedUatCaseSet | NumberedUatCaseSet:
    """Return two stub UAT cases, tagged with the agent ID."""
    cases = [
        GeneratedUatCase(
            title="Log in with valid credentials",
            description=(
                f"{config.id}: enter a registered email and password; "
                "the home page opens"
            ),
        ),
        GeneratedUatCase(
            title="Reject a wrong password",
            description=(
                f"{config.id}: enter a wrong password; an error shows and login fails"
            ),
        ),
    ]
    if output_type is NumberedUatCaseSet:
        return NumberedUatCaseSet(
            cases=[
                NumberedUatCase(title_number=number, description=case.description)
                for number, case in enumerate(cases, start=1)
            ]
        )
    return GeneratedUatCaseSet(cases=cases)


def _stub_criteria(config: GenerationAgentConfig) -> GeneratedCriteriaSet:
    """Return two criteria every agent shares, and one unique to `config`."""
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


def _build_model(config: GenerationAgentConfig, output_type: GenerationOutput) -> Model:
    """Build the model for a roster entry, stubbing the `test` provider."""
    if config.provider == "test":
        from pydantic_ai.models.test import TestModel

        return TestModel(custom_output_args=_stub_output(config, output_type))

    return build_model(config)


def build_agent(
    config: GenerationAgentConfig,
    output_type: GenerationOutput = GeneratedCriteriaSet,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    with_context: bool = True,
) -> GenerationAgent:
    """Build a generation agent for one roster entry.

    Args:
        config: The roster entry.
        output_type: The `*Set` model the agent returns.
        system_prompt: The base system prompt.
        with_context: Whether to append project context and reviewer feedback.

    Raises:
        ModelBuildError: The provider is unknown or its API key is missing.
    """
    agent: GenerationAgent = Agent(
        _build_model(config, output_type),
        output_type=with_output_mode(output_type, config.output_mode),
        deps_type=GenerationDeps,
        system_prompt=system_prompt,
        model_settings=config.model_settings(),
        retries=2,
    )

    if not with_context:
        return agent

    @agent.system_prompt
    def inject_project_context(run_context: RunContext[GenerationDeps]) -> str:
        """Append the retrieved project context (backlog R11)."""
        return run_context.deps.context_block()

    @agent.system_prompt
    def inject_reviewer_feedback(run_context: RunContext[GenerationDeps]) -> str:
        """Append reviewer feedback when regenerating (backlog R20)."""
        return run_context.deps.feedback_block() or ""

    return agent
