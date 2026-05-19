"""
Generation Agent — one participant in the ensemble.

Each agent is a PydanticAI `Agent` bound to a specific provider/model via
its AgentConfig. PydanticAI enforces the output schema (AgentOutput) at the
provider level with no manual JSON parsing & no schema-in-prompt.

The orchestrator spawns all agents concurrently via asyncio.gather.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from .context import RunContext
from .models import (
    AcceptanceCriteria,
    AgentConfig,
    AgentResult,
    GenerationStatus,
    Provider,
    UserStory,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


# Output schema enforced by PydanticAI
#
# This is the *wire shape* the LLM must produce. It is intentionally separate
# from the storage model (models.AcceptanceCriteria) so the LLM contract and
# the persistence/transport contract can evolve independently.

class ScenarioSpec(BaseModel):
    scenario_title: str = Field(description="Short title summarising the scenario")
    given: list[str]    = Field(description="Preconditions or initial state",        min_length=1)
    when:  list[str]    = Field(description="The action(s) or event(s) under test",  min_length=1)
    then:  list[str]    = Field(description="Expected, observable outcomes",         min_length=1)


class AgentOutput(BaseModel):
    scenarios: list[ScenarioSpec] = Field(
        description="Gherkin scenarios covering happy path and key edge/error cases",
        min_length=4,
        max_length=6,
    )


# -----------------------------------------------
# Provider mapping for PydanticAI model strings
# -----------------------------------------------

_PROVIDER_PREFIX = {
    Provider.OPENAI: "openai-chat",  # Chat Completions API
    Provider.GEMINI: "google",       # Google AI Studio / Gemini API
}


def _model_string(agent_cfg: AgentConfig) -> str:
    return f"{_PROVIDER_PREFIX[agent_cfg.provider]}:{agent_cfg.model}"


# ---------------------
# Public entry point
# ---------------------

async def run_generation_agent(
    agent_cfg: AgentConfig,
    story: UserStory,
    context: RunContext,
) -> AgentResult:
    """
    Execute a single Generation Agent asynchronously.

    Args:
        agent_cfg: Per-agent config (provider, model, temperature, etc.)
        story:     The UserStory to generate criteria for.
        context:   Injected RunContext (vector store, run_id, etc.)

    Returns:
        AgentResult containing parsed criteria or an error payload.
    """
    label = f"Agent #{agent_cfg.agent_id} [{agent_cfg.provider.value} / {agent_cfg.model}]"
    logger.info("%s starting (temp=%.2f)", label, agent_cfg.temperature)

    t_start = time.monotonic()

    rag_snippets = context.vector_store.similarity_search(story.description, k=3)
    extra_context = story.additional_context or ""
    if rag_snippets:
        extra_context += "\n\n## Retrieved Context\n" + "\n".join(rag_snippets)

    user_prompt = build_user_prompt(
        story_title=story.title,
        story_description=story.description,
        additional_context=extra_context or None,
        ui_context=story.ui_context,
    )

    agent: Agent[None, AgentOutput] = Agent(
        _model_string(agent_cfg),
        output_type=AgentOutput,
        system_prompt=SYSTEM_PROMPT,
        output_retries=3,
        model_settings=ModelSettings(
            temperature=agent_cfg.temperature,
            max_tokens=agent_cfg.max_tokens,
        ),
    )

    try:
        result = await agent.run(user_prompt)
        latency_ms = (time.monotonic() - t_start) * 1000

        criteria = [
            AcceptanceCriteria(
                scenario_title=s.scenario_title,
                given=s.given,
                when=s.when,
                then=s.then,
                raw_text=s.model_dump_json(),
            )
            for s in result.output.scenarios
        ]
        logger.info("%s done: %d criteria in %.0fms", label, len(criteria), latency_ms)

        return AgentResult(
            agent_id=agent_cfg.agent_id,
            user_story_id=story.id,
            status=GenerationStatus.SUCCESS,
            criteria=criteria,
            raw_response=result.output.model_dump_json(),
            temperature_used=agent_cfg.temperature,
            latency_ms=latency_ms,
        )

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - t_start) * 1000
        logger.error("%s failed after %.0fms: %s", label, latency_ms, exc)
        return AgentResult(
            agent_id=agent_cfg.agent_id,
            user_story_id=story.id,
            status=GenerationStatus.FAILED,
            error_message=str(exc),
            temperature_used=agent_cfg.temperature,
            latency_ms=latency_ms,
        )
