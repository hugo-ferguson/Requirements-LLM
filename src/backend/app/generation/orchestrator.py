"""The ensemble orchestrator: fan out to every enabled agent at once.

This is the core of the generation layer. Running the agents concurrently means
total latency tracks the slowest agent rather than the sum of all of them
(backlog R10), which is what keeps the run inside the 60 second budget in R6.

There is deliberately no agent framework here — `asyncio.gather` over
async-native PydanticAI agents is the whole mechanism.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.generation.agents import AgentBuildError, build_agent
from app.generation.config import GenerationAgentConfig, get_roster
from app.generation.models import AgentResult, EnsembleResult, GeneratedCriterion
from app.generation.prompts import GenerationDeps, build_user_prompt

logger = logging.getLogger(__name__)


async def _run_one(
    config: GenerationAgentConfig,
    deps: GenerationDeps,
    *,
    max_criteria: int,
    timeout_seconds: float,
) -> AgentResult:
    """Run a single agent, converting every failure mode into a result object.

    Nothing raises out of here. A dead agent must degrade the ensemble, not
    fail the request — the same convention the voting layer uses.
    """
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        agent = build_agent(config)
        prompt = build_user_prompt(deps, max_criteria)
        run = await asyncio.wait_for(agent.run(prompt, deps=deps), timeout=timeout_seconds)
    except AgentBuildError as error:
        logger.warning("Generation agent %s not available: %s",
                       config.id, error)
        return AgentResult(
            agent_id=config.id,
            provider=config.provider,
            model=config.model,
            duration_ms=elapsed_ms(),
            error=str(error),
        )
    except asyncio.TimeoutError:
        logger.warning("Generation agent %s timed out after %ss",
                       config.id, timeout_seconds)
        return AgentResult(
            agent_id=config.id,
            provider=config.provider,
            model=config.model,
            duration_ms=elapsed_ms(),
            error=f"Timed out after {timeout_seconds}s",
        )
    except Exception as error:  # noqa: BLE001 - one bad agent must not fail the run
        logger.exception("Generation agent %s failed", config.id)
        return AgentResult(
            agent_id=config.id,
            provider=config.provider,
            model=config.model,
            duration_ms=elapsed_ms(),
            error=f"{error.__class__.__name__}: {error}",
        )

    return AgentResult(
        agent_id=config.id,
        provider=config.provider,
        model=config.model,
        criteria=list(run.output.criteria)[:max_criteria],
        duration_ms=elapsed_ms(),
    )


async def run_ensemble(
    deps: GenerationDeps,
    *,
    agents: list[GenerationAgentConfig] | None = None,
    max_criteria: int = 8,
    timeout_seconds: float = 90.0,
) -> EnsembleResult:
    """Dispatch the same story to every enabled agent simultaneously."""
    roster = agents if agents is not None else get_roster().enabled_agents()

    results = await asyncio.gather(
        *(
            _run_one(
                config,
                deps,
                max_criteria=max_criteria,
                timeout_seconds=timeout_seconds,
            )
            for config in roster
        )
    )

    return EnsembleResult(
        prompt=build_user_prompt(deps, max_criteria),
        results=list(results),
    )


def pool_criteria(ensemble: EnsembleResult) -> list[tuple[str, GeneratedCriterion]]:
    """Flatten every successful agent's criteria into one candidate pool.

    Returns `(agent_id, criterion)` pairs so provenance survives into logging
    and evaluation, even though the voting layer itself never sees the labels.
    Exact duplicates across agents are dropped — three models independently
    producing the same criterion should appear once, not three times.
    """
    seen: set[str] = set()
    pooled: list[tuple[str, GeneratedCriterion]] = []

    for result in ensemble.successful:
        for criterion in result.criteria:
            key = criterion.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            pooled.append((result.agent_id, criterion))

    return pooled
