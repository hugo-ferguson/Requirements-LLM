"""Run the generation agents concurrently (R10)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from app.generation.agents import GenerationOutput, build_agent
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
from app.llm_config import GenerationAgentConfig, get_models_config
from llm.spec import ModelBuildError

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
    """Run one agent, returning failures as a result instead of raising.

    Passing `anchor_titles` makes it a pass-2 run that answers by title number.
    """
    if anchor_titles is None:
        make_agent = partial(build_agent, config)
    else:
        make_agent = partial(build_agent, config, output_type=NumberedCriteriaSet)

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
    """What one agent run produced: its output, or why it failed."""

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
    """Build and run one agent, returning any failure as `error`."""
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        agent = make_agent()
        agent_result = await asyncio.wait_for(
            agent.run(prompt, deps=deps), timeout=timeout_seconds
        )
    except ModelBuildError as error:
        logger.warning("Generation agent %s not available: %s", config.id, error)
        return _AgentRun(error=str(error), duration_ms=elapsed_ms())
    except TimeoutError:
        logger.warning(
            "Generation agent %s timed out after %ss", config.id, timeout_seconds
        )
        return _AgentRun(
            error=f"Timed out after {timeout_seconds}s", duration_ms=elapsed_ms()
        )
    except Exception as error:  # One bad agent mustn't fail the run.
        logger.exception("Generation agent %s failed", config.id)
        return _AgentRun(
            error=f"{error.__class__.__name__}: {error}", duration_ms=elapsed_ms()
        )
    return _AgentRun(output=agent_result.output, duration_ms=elapsed_ms())


def _attach_anchor_titles(
    agent_id: str, output: NumberedCriteriaSet, titles: list[str]
) -> list[GeneratedCriterion]:
    """Turn numbered pass-2 answers into titled criteria.

    Drops unknown or repeated numbers.
    """
    criteria: list[GeneratedCriterion] = []
    answered: set[int] = set()
    for answer in output.criteria:
        number = answer.title_number
        if not 1 <= number <= len(titles) or number in answered:
            continue
        answered.add(number)
        criteria.append(
            GeneratedCriterion(
                title=titles[number - 1],
                given=answer.given,
                when=answer.when,
                then=answer.then,
            )
        )

    dropped = len(output.criteria) - len(criteria)
    if dropped:
        logger.info(
            "agent=%s gave %d answer(s) with an unknown or repeated title number; "
            "dropped",
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
    """Send the same story to every agent at once.

    Args:
        deps: The story, context and any reviewer feedback.
        agents: The roster. Defaults to the enabled agents.
        max_criteria: Maximum criteria kept per agent.
        timeout_seconds: Timeout for each agent.

    Returns:
        Every agent's result, failed ones included.
    """
    roster = agents if agents is not None else get_models_config().enabled_agents()

    results = await asyncio.gather(
        *(
            _run_one(
                agent_config,
                deps,
                max_criteria=max_criteria,
                timeout_seconds=timeout_seconds,
            )
            for agent_config in roster
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
    """Run pass 1 (titles) on the first agent, then pass 2 on the rest.

    Args:
        deps: The story, context and any reviewer feedback.
        agents: The roster, anchor first. Defaults to the enabled agents.
        max_criteria: Maximum criteria kept per agent.
        timeout_seconds: Timeout for each agent.

    Returns:
        Every agent's result, and one group per title. No groups if the
        anchor produced nothing.
    """
    roster = agents if agents is not None else get_models_config().enabled_agents()
    if not roster:
        return EnsembleResult(
            prompt=build_user_prompt(deps, max_criteria), results=[]
        ), []

    anchor_config, *follower_configs = roster

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
        # Nothing to anchor against. The caller falls back to `run_ensemble`.
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
                agent_config,
                deps,
                max_criteria=max_criteria,
                timeout_seconds=timeout_seconds,
                prompt=descriptions_prompt,
                anchor_titles=titles,
            )
            for agent_config in follower_configs
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
    """Group every agent's criteria by anchor title, in the anchor's order.

    Drops follower criteria that match no anchor title.
    """
    groups: dict[str, CandidateGroup] = {}
    for criterion in anchor.criteria:
        key = title_key(criterion.title)
        if key in groups:
            # Keep the first of two matching anchor titles.
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
            1 for criterion in follower.criteria if title_key(criterion.title) in groups
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
    """Pool every successful agent's criteria, dropping exact duplicates.

    Returns:
        `(agent_id, criterion)` pairs.
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


# UAT cases use the same two passes as ACs, once per accepted AC.


def _uat_agent(config: GenerationAgentConfig, output_type: GenerationOutput):
    """Build a generation agent with the UAT prompt and no story context."""
    return build_agent(
        config,
        output_type=output_type,
        system_prompt=UAT_SYSTEM_PROMPT,
        with_context=False,
    )


def _uat_result(
    config: GenerationAgentConfig,
    run: _AgentRun,
    cases: list[GeneratedUatCase] | None = None,
) -> UatAgentResult:
    """Package one agent run as a UAT result, failed or not."""
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
    """Generate UAT cases for one acceptance criterion, in two passes.

    If the anchor fails, the next agent in the roster takes its place.

    Args:
        acceptance_criterion: The AC as text.
        agents: The roster. Defaults to the enabled agents.
        timeout_seconds: Timeout for each agent.

    Returns:
        Every agent's result, and one group per case title. No groups only
        if every agent failed.
    """
    roster = agents if agents is not None else get_models_config().enabled_agents()
    deps = GenerationDeps(user_story=acceptance_criterion)
    results: list[UatAgentResult] = []

    anchor: UatAgentResult | None = None
    remaining = list(roster)
    while remaining and anchor is None:
        agent_config = remaining.pop(0)
        run = await _run_agent(
            agent_config,
            partial(_uat_agent, agent_config, GeneratedUatCaseSet),
            build_uat_cases_prompt(acceptance_criterion),
            deps,
            timeout_seconds,
        )
        result = _uat_result(
            agent_config, run, list(run.output.cases) if run.output else None
        )
        results.append(result)
        if result.ok and result.cases:
            anchor = result

    if anchor is None:
        return results, []

    titles = [case.title for case in anchor.cases]
    follower_runs = await asyncio.gather(
        *(
            _run_agent(
                agent_config,
                partial(_uat_agent, agent_config, NumberedUatCaseSet),
                build_uat_descriptions_prompt(acceptance_criterion, titles),
                deps,
                timeout_seconds,
            )
            for agent_config in remaining
        )
    )
    followers = [
        _uat_result(
            agent_config,
            run,
            _attach_uat_titles(agent_config.id, run.output, titles)
            if run.output
            else None,
        )
        for agent_config, run in zip(remaining, follower_runs, strict=True)
    ]
    results.extend(followers)

    groups: dict[str, UatCandidateGroup] = {}
    for case in anchor.cases:
        key = title_key(case.title)
        if key not in groups:
            groups[key] = UatCandidateGroup(
                title=case.title,
                candidates=[UatCandidate(agent_id=anchor.agent_id, case=case)],
            )
    for follower in followers:
        for case in follower.cases:
            group = groups.get(title_key(case.title))
            if group is not None and all(
                candidate.agent_id != follower.agent_id
                for candidate in group.candidates
            ):
                group.candidates.append(
                    UatCandidate(agent_id=follower.agent_id, case=case)
                )

    return results, list(groups.values())


def _attach_uat_titles(
    agent_id: str, output: NumberedUatCaseSet, titles: list[str]
) -> list[GeneratedUatCase]:
    """Give UAT answers their titles, like `_attach_anchor_titles`."""
    cases: list[GeneratedUatCase] = []
    answered: set[int] = set()
    for answer in output.cases:
        number = answer.title_number
        if not 1 <= number <= len(titles) or number in answered:
            continue
        answered.add(number)
        cases.append(
            GeneratedUatCase(title=titles[number - 1], description=answer.description)
        )

    dropped = len(output.cases) - len(cases)
    if dropped:
        logger.info(
            "agent=%s gave %d UAT answer(s) with an unknown or repeated number; "
            "dropped",
            agent_id,
            dropped,
        )
    return cases
