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
from app.generation.models import (
    AgentResult,
    Candidate,
    CandidateGroup,
    EnsembleResult,
    GeneratedCriterion,
    title_key,
)
from app.generation.prompts import (
    GenerationDeps,
    build_descriptions_prompt,
    build_titles_prompt,
    build_user_prompt,
)

logger = logging.getLogger(__name__)


async def _run_one(
    config: GenerationAgentConfig,
    deps: GenerationDeps,
    *,
    max_criteria: int,
    timeout_seconds: float,
    prompt: str | None = None,
) -> AgentResult:
    """Run a single agent, converting every failure mode into a result object.

    Nothing raises out of here. A dead agent must degrade the ensemble, not
    fail the request — the same convention the voting layer uses.

    `prompt` overrides the default user message so the title-anchored passes
    can reuse this same failure handling instead of duplicating it.
    """
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        agent = build_agent(config)
        prompt = prompt if prompt is not None else build_user_prompt(deps, max_criteria)
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


async def run_title_anchored_ensemble(
    deps: GenerationDeps,
    *,
    agents: list[GenerationAgentConfig] | None = None,
    max_criteria: int = 8,
    timeout_seconds: float = 90.0,
) -> tuple[EnsembleResult, list[CandidateGroup]]:
    """Two-pass generation: one agent fixes the titles, the rest fill them in.

    Pass 1 runs a single nominated agent. Its titles become the grouping key.
    Pass 2 runs every *other* agent concurrently against those titles, so each
    group holds one candidate per agent describing the same behaviour — which
    is what lets the voting layer compare like against like instead of scoring
    an arbitrary union (see the review notes on duplicate criteria).

    Latency is pass 1 plus the slowest agent in pass 2, rather than the slowest
    agent overall. That is the inherent cost of anchoring, and it is why the
    caller can still fall back to `run_ensemble`.

    Returns the pass-1 ensemble alongside the groups, because the caller still
    needs `EnsembleResult.prompt` for scoring and `failed` for error reporting.
    """
    roster = agents if agents is not None else get_roster().enabled_agents()
    if not roster:
        return EnsembleResult(prompt=build_user_prompt(deps, max_criteria), results=[]), []

    anchor_config, *rest = roster

    anchor = await _run_one(
        anchor_config,
        deps,
        max_criteria=max_criteria,
        timeout_seconds=timeout_seconds,
        prompt=build_titles_prompt(deps, max_criteria),
    )

    ensemble = EnsembleResult(
        prompt=build_user_prompt(deps, max_criteria),
        results=[anchor],
    )

    if not anchor.ok or not anchor.criteria:
        # Nothing to anchor against. The caller falls back to the flat
        # ensemble rather than returning an empty run.
        logger.warning(
            "title-anchor agent %s produced no criteria (%s); "
            "title-anchored generation cannot proceed",
            anchor_config.id,
            anchor.error or "empty result",
        )
        return ensemble, []

    titles = [criterion.title for criterion in anchor.criteria]
    descriptions_prompt = build_descriptions_prompt(deps, titles)

    followers = await asyncio.gather(
        *(
            _run_one(
                config,
                deps,
                max_criteria=max_criteria,
                timeout_seconds=timeout_seconds,
                prompt=descriptions_prompt,
            )
            for config in rest
        )
    )

    ensemble = EnsembleResult(
        prompt=build_user_prompt(deps, max_criteria),
        results=[anchor, *followers],
    )

    return ensemble, group_by_title(anchor, list(followers))


def group_by_title(
    anchor: AgentResult, followers: list[AgentResult]
) -> list[CandidateGroup]:
    """Collect every agent's criteria into one group per anchor title.

    Group order follows the anchor's title order, so the anchor agent decides
    both the content and the sequence of the final list. A follower criterion
    whose title matches nothing is dropped rather than promoted to its own
    group: pass 2 asked for a fixed set, so an unmatched title means the model
    ignored the instruction, and letting it through would reintroduce exactly
    the ungrouped duplicates this design exists to prevent.
    """
    groups: dict[str, CandidateGroup] = {}
    for criterion in anchor.criteria:
        key = title_key(criterion.title)
        if key in groups:
            # The anchor itself produced two titles that normalise the same.
            # Keep the first; the second would create a duplicate group.
            continue
        groups[key] = CandidateGroup(
            title=criterion.title,
            candidates=[Candidate(agent_id=anchor.agent_id, criterion=criterion)],
        )

    for follower in followers:
        seen: set[str] = set()
        for criterion in follower.criteria:
            key = title_key(criterion.title)
            group = groups.get(key)
            if group is None or key in seen:
                continue
            seen.add(key)
            group.candidates.append(
                Candidate(agent_id=follower.agent_id, criterion=criterion)
            )

    for follower in followers:
        matched = sum(
            1 for c in follower.criteria if title_key(c.title) in groups
        )
        if follower.ok and matched < len(follower.criteria):
            logger.info(
                "agent=%s returned %d criteria, %d matched an anchor title",
                follower.agent_id,
                len(follower.criteria),
                matched,
            )

    return list(groups.values())


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
