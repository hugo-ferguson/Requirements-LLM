"""
Ensemble Orchestrator

Responsibilities:
  1. Spawn N Generation Agents concurrently via asyncio.gather
  2. Collect all AgentResults into an EnsembleResult
  3. Hand the EnsembleResult off to the Voting Layer (next pipeline stage)

The orchestrator does NOT decide which output is best — that belongs to
the Voting Layer. Its only job is reliable parallel execution + collection.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from .agent import run_generation_agent
from .context import RunContext
from .models import AgentResult, EnsembleResult, GenerationStatus, UserStory

logger = logging.getLogger(__name__)


class EnsembleOrchestrator:
    """
    Coordinates parallel generation across N agents.

    Usage:
        config = EnsembleConfig(agents=[
            AgentConfig(agent_id=0, provider=Provider.OPENAI, model="gpt-4o-mini"),
            AgentConfig(agent_id=1, provider=Provider.GEMINI, model="gemini-2.5-flash"),
        ])
        context = RunContext.build(config=config)
        orch = EnsembleOrchestrator(context)
        result = await orch.run(story)
        # result is an EnsembleResult ready for the Voting Layer
    """

    def __init__(self, context: RunContext) -> None:
        self.context = context

    async def run(
        self,
        story: UserStory,
        run_id: str | None = None,
    ) -> EnsembleResult:
        """
        Launch all N agents concurrently and collect their results.

        Args:
            story:  The user story to generate acceptance criteria for.
            run_id: Optional correlation ID (auto-generated if not provided).

        Returns:
            EnsembleResult — all agent outputs bundled for the Voting Layer.
        """
        run_id = run_id or str(uuid.uuid4())
        n = self.context.config.num_agents

        logger.info(
            "Ensemble run %s starting: %d agents for story '%s'",
            run_id[:8], n, story.title,
        )

        t_start = time.monotonic()

        # Core: launch all agents simultaneously via asyncio.gather.
        # return_exceptions=True ensures one agent failure never cancels
        # the rest — a partial ensemble result is still useful for voting.
        tasks = [
            run_generation_agent(agent_cfg=self.context.config.agents[i], story=story, context=self.context)
            for i in range(n)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        total_latency_ms = (time.monotonic() - t_start) * 1000

        # Unwrap any unexpected exceptions that slipped past agent-level handling
        agent_results: list[AgentResult] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.error("Unhandled exception from agent #%d: %s", i, r)
                agent_results.append(
                    AgentResult(
                        agent_id=i,
                        user_story_id=story.id,
                        status=GenerationStatus.FAILED,
                        error_message=f"Unhandled: {r}",
                    )
                )
            else:
                agent_results.append(r)

        result = EnsembleResult(
            run_id=run_id,
            user_story=story,
            config=self.context.config,
            agent_results=agent_results,
            total_latency_ms=total_latency_ms,
        )

        logger.info(
            "Ensemble run %s complete: %s | success_rate=%.0f%%",
            run_id[:8],
            result.summary(),
            result.success_rate * 100,
        )

        return result