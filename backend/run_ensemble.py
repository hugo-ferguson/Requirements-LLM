"""Barebones dev runner for the Ensemble Generation Layer."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ensemble_layer import (
    AgentConfig,
    EnsembleConfig,
    EnsembleOrchestrator,
    Provider,
    RunContext,
    UserStory,
)

MOCK_INPUT = Path(__file__).parent / "mock_stories.json"

AGENTS = [
    AgentConfig(agent_id=0, provider=Provider.OPENAI, model="gpt-4o-mini", temperature=0.7),
    AgentConfig(agent_id=1, provider=Provider.GEMINI, model="gemini-2.5-flash", temperature=0.7),
]


async def run(args: argparse.Namespace) -> None:
    raw = json.loads(MOCK_INPUT.read_text())
    stories = [UserStory(**s) for s in raw["user_stories"]]

    if args.story is not None:
        stories = [stories[args.story]]

    context = RunContext.build(config=EnsembleConfig(agents=AGENTS))
    orchestrator = EnsembleOrchestrator(context)

    results = [await orchestrator.run(story) for story in stories]
    exclude = (
        {"agent_results": {"__all__": {"raw_response": True, "criteria": {"__all__": {"raw_text": True}}}}}
        if args.compact else None
    )
    payload = [r.model_dump(mode="json", exclude=exclude) for r in results]
    output = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--story", type=int, default=None)
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Write JSON to this file instead of stdout")
    p.add_argument("--compact", action="store_true",
                   help="Exclude raw_response and per-criterion raw_text from output")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
