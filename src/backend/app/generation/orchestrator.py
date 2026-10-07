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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.generation.agents import AgentBuildError, build_agent
from app.llm_config import GenerationAgentConfig, get_models_config
from app.generation.models import (
    AgentResult,
    Candidate,
    CandidateGroup,
    EnsembleResult,
    GeneratedCriterion,
    GeneratedUatCase,
    GeneratedUatCaseSet,
    NumberedCriteriaSet,
    NumberedUatCaseSet,
    UatAgentResult,
    UatCandidate,
    UatCandidateGroup,
    title_key,
)
from app.generation.prompts import (
    UAT_SYSTEM_PROMPT,
    GenerationDeps,
    build_descriptions_prompt,
    build_titles_prompt,
    build_uat_cases_prompt,
    build_uat_descriptions_prompt,
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
    anchor_titles: list[str] | None = None,
) -> AgentResult:
    """Run a single agent, converting every failure mode into a result object.

    Nothing raises out of here. A dead agent must degrade the ensemble, not
    fail the request — the same convention the voting layer uses.

    `prompt` overrides the default user message so the title-anchored passes
    can reuse this same failure handling instead of duplicating it. Passing
    `anchor_titles` makes this a pass-2 run: the agent answers by title
    number and the anchor's exact titles are attached here.
    """
    if anchor_titles is None:
        make_agent = lambda: build_agent(config)  # noqa: E731
    else:
        make_agent = lambda: build_agent(config, output_type=NumberedCriteriaSet)  # noqa: E731
    run = await _run_agent(
        config,
        make_agent,
        prompt if prompt is not None else build_user_prompt(deps, max_criteria),
        deps,
        timeout_seconds,
    )
    if run.error is not None:
        return AgentResult(
            agent_id=config.id,
            provider=config.provider,
            model=config.model,
            duration_ms=run.duration_ms,
            error=run.error,
        )

    if anchor_titles is None:
        criteria = list(run.output.criteria)
    else:
        criteria = _attach_anchor_titles(config.id, run.output, anchor_titles)

    return AgentResult(
        agent_id=config.id,
        provider=config.provider,
        model=config.model,
        criteria=criteria[:max_criteria],
        duration_ms=run.duration_ms,
    )


@dataclass
class _AgentRun:
    output: Any = None
    error: str | None = None
    duration_ms: int = 0


async def _run_agent(
    config: GenerationAgentConfig,
    make_agent: Callable[[], Any],
    prompt: str,
    deps: GenerationDeps,
    timeout_seconds: float,
) -> _AgentRun:
    """Build and run one agent; every failure comes back as `error`, never raised."""
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        agent = make_agent()
        run = await asyncio.wait_for(agent.run(prompt, deps=deps), timeout=timeout_seconds)
    except AgentBuildError as error:
        logger.warning("Generation agent %s not available: %s", config.id, error)
        return _AgentRun(error=str(error), duration_ms=elapsed_ms())
    except asyncio.TimeoutError:
        logger.warning("Generation agent %s timed out after %ss", config.id, timeout_seconds)
        return _AgentRun(error=f"Timed out after {timeout_seconds}s", duration_ms=elapsed_ms())
    except Exception as error:  # noqa: BLE001 - one bad agent must not fail the run
        logger.exception("Generation agent %s failed", config.id)
        return _AgentRun(
            error=f"{error.__class__.__name__}: {error}", duration_ms=elapsed_ms()
        )
    return _AgentRun(output=run.output, duration_ms=elapsed_ms())


def _attach_anchor_titles(
    agent_id: str, output: NumberedCriteriaSet, titles: list[str]
) -> list[GeneratedCriterion]:
    """Turn numbered pass-2 answers into criteria carrying the anchor's titles.

    A number outside the list, or a second answer to the same number, is
    dropped: there is no title to attach, or the group already has this
    agent's answer.
    """
    criteria: list[GeneratedCriterion] = []
    answered: set[int] = set()
    for item in output.criteria:
        number = item.title_number
        if not 1 <= number <= len(titles) or number in answered:
            continue
        answered.add(number)
        criteria.append(
            GeneratedCriterion(
                title=titles[number - 1], given=item.given, when=item.when, then=item.then
            )
        )

    dropped = len(output.criteria) - len(criteria)
    if dropped:
        logger.info(
            "agent=%s gave %d answer(s) with an unknown or repeated title number; dropped",
            agent_id,
            dropped,
        )
    return criteria


async def run_ensemble(
    deps: GenerationDeps,
    *,
    agents: list[GenerationAgentConfig] | None = None,
    max_criteria: int = 8,
    timeout_seconds: float = 90.0,
) -> EnsembleResult:
    """Dispatch the same story to every enabled agent simultaneously."""
    roster = agents if agents is not None else get_models_config().enabled_agents()

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
    roster = agents if agents is not None else get_models_config().enabled_agents()
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
                anchor_titles=titles,
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


# --- UAT test cases ----------------------------------------------------------


def _uat_agent(config: GenerationAgentConfig, output_type):
    return build_agent(
        config, output_type=output_type, system_prompt=UAT_SYSTEM_PROMPT, with_context=False
    )


def _uat_result(config: GenerationAgentConfig, run: _AgentRun, cases=None) -> UatAgentResult:
    return UatAgentResult(
        agent_id=config.id,
        provider=config.provider,
        model=config.model,
        cases=cases or [],
        duration_ms=run.duration_ms,
        error=run.error,
    )


async def run_uat_ensemble(
    acceptance_criterion: str,
    *,
    agents: list[GenerationAgentConfig] | None = None,
    timeout_seconds: float = 90.0,
) -> tuple[list[UatAgentResult], list[UatCandidateGroup]]:
    """Two-pass UAT generation for one acceptance criterion.

    The first agent that succeeds anchors the test case titles; every other
    agent then writes its own description for each, by case number. Unlike AC
    generation there is no flat fallback: if the preferred anchor fails, the
    next agent in roster order anchors instead, so one dead model still
    leaves a usable set. Returns every agent's result (for error reporting)
    and one group per case title — empty only when every agent failed.
    """
    roster = agents if agents is not None else get_models_config().enabled_agents()
    deps = GenerationDeps(user_story=acceptance_criterion)
    results: list[UatAgentResult] = []

    anchor: UatAgentResult | None = None
    remaining = list(roster)
    while remaining and anchor is None:
        config = remaining.pop(0)
        run = await _run_agent(
            config,
            lambda config=config: _uat_agent(config, GeneratedUatCaseSet),
            build_uat_cases_prompt(acceptance_criterion),
            deps,
            timeout_seconds,
        )
        result = _uat_result(config, run, list(run.output.cases) if run.output else None)
        results.append(result)
        if result.ok and result.cases:
            anchor = result

    if anchor is None:
        return results, []

    titles = [case.title for case in anchor.cases]
    follower_runs = await asyncio.gather(
        *(
            _run_agent(
                config,
                lambda config=config: _uat_agent(config, NumberedUatCaseSet),
                build_uat_descriptions_prompt(acceptance_criterion, titles),
                deps,
                timeout_seconds,
            )
            for config in remaining
        )
    )
    followers = [
        _uat_result(
            config,
            run,
            _attach_uat_titles(config.id, run.output, titles) if run.output else None,
        )
        for config, run in zip(remaining, follower_runs, strict=True)
    ]
    results.extend(followers)

    groups: dict[str, UatCandidateGroup] = {}
    for case in anchor.cases:
        key = title_key(case.title)
        if key not in groups:
            groups[key] = UatCandidateGroup(
                title=case.title, candidates=[UatCandidate(agent_id=anchor.agent_id, case=case)]
            )
    for follower in followers:
        for case in follower.cases:
            group = groups.get(title_key(case.title))
            if group is not None and all(c.agent_id != follower.agent_id for c in group.candidates):
                group.candidates.append(UatCandidate(agent_id=follower.agent_id, case=case))

    return results, list(groups.values())


def _attach_uat_titles(
    agent_id: str, output: NumberedUatCaseSet, titles: list[str]
) -> list[GeneratedUatCase]:
    """UAT counterpart of `_attach_anchor_titles`, with the same drop rules."""
    cases: list[GeneratedUatCase] = []
    answered: set[int] = set()
    for item in output.cases:
        number = item.title_number
        if not 1 <= number <= len(titles) or number in answered:
            continue
        answered.add(number)
        cases.append(GeneratedUatCase(title=titles[number - 1], description=item.description))

    dropped = len(output.cases) - len(cases)
    if dropped:
        logger.info(
            "agent=%s gave %d UAT answer(s) with an unknown or repeated number; dropped",
            agent_id,
            dropped,
        )
    return cases
