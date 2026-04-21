"""
PydanticAI Proof-of-Concept Demonstration

This file is a self-contained spike intended to walk team members through
the core PydanticAI concepts that underpin the main Project architecture:

  CONCEPT 1 — Structured output schema (output_type)
              PydanticAI enforces a Pydantic model on every LLM response.
              If the LLM returns malformed output, it retries automatically.
              This replaces any terminal "validate Gherkin" step — validation
              happens inside the agent, before the output ever leaves it.

  CONCEPT 2 — RunContext and dependency injection
              RunContext is PydanticAI's mechanism for passing shared state
              (project config, domain context, DB clients) into an agent run.
              It is scoped to ONE agent run. It is NOT an inter-agent
              orchestrator — that role belongs to the Python orchestrator below.

  CONCEPT 3 — Multiple agents, one schema
              Swapping the model string is all that changes between agents.
              output_type, system_prompt, and deps_type stay identical.
              This is the ensemble: same task, different model, same contract.

  CONCEPT 4 — Async parallel dispatch (asyncio.gather)
              The Python orchestrator fans out to all agents simultaneously
              using asyncio.gather. No framework required — this is the core
              of the Project 30 pipeline.

  CONCEPT 5 — Voting
              Each agent returns a validated AcceptanceCriteria object.
              A plain Python function selects the winner. This spike uses a
              naive count-based vote; production would use semantic scoring.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONCEPT 1 — OUTPUT SCHEMA
#
# AcceptanceCriterion maps directly to one Gherkin scenario.
# AcceptanceCriteria is the full set returned by a single agent run.
#
# The Field descriptions are passed to the LLM as part of the prompt — they
# guide generation without requiring complex prompt engineering.
#
# min_length=1 and max_length=3 are validated by Pydantic before the result
# is returned. If violated, PydanticAI asks the LLM to try again.
# =============================================================================

class AcceptanceCriterion(BaseModel):
    given: str = Field(description="Precondition or initial system/user state")
    when: str  = Field(description="The specific action or event being tested")
    then: str  = Field(description="The expected, observable outcome")


class AcceptanceCriteria(BaseModel):
    user_story_summary: str = Field(
        description="One-sentence summary of the user story being addressed"
    )
    criteria: list[AcceptanceCriterion] = Field(
        description="Gherkin-style acceptance criteria, one per testable behaviour",
        min_length=1,
        max_length=3,
    )


# =============================================================================
# CONCEPT 2 — DEPENDENCIES (RunContext)
#
# ProjectDeps holds anything the agent needs at runtime that isn't part of
# the user prompt — project name, domain docs, retrieved RAG context, etc.
#
# This is injected into the agent via deps= at call time, not hardcoded into
# the agent definition. That means the same agent instance can serve different
# projects by changing the deps object.
# =============================================================================

@dataclass
class ProjectDeps:
    project_name:   str
    domain_context: str


# =============================================================================
# CONCEPT 3 — AGENT FACTORY
#
# create_agent() creates a PydanticAI Agent bound to a specific LLM.
# The output_type, deps_type, and base system_prompt are shared across all
# three agents in the ensemble — only the model string changes.
#
# The @agent.system_prompt decorator injects runtime context from RunContext
# as an additional system message appended to the base prompt. This is
# PydanticAI's idiomatic way to use deps in prompt construction.
# =============================================================================

BASE_SYSTEM_PROMPT = """\
You are an expert business analyst writing acceptance criteria for software features.
Given a user story and project domain context, produce clear, atomic, testable
Gherkin acceptance criteria in Given / When / Then format.
Each criterion should cover one distinct user-observable behaviour.
Avoid implementation detail. Avoid vague language like "the system works correctly".
"""


def create_agent(model_string: str) -> Agent[ProjectDeps, AcceptanceCriteria]:
    """
    Returns a PydanticAI Agent configured for acceptance criteria generation.
    Swap model_string to change the underlying LLM — everything else stays fixed.
    """
    agent: Agent[ProjectDeps, AcceptanceCriteria] = Agent(
        model_string,
        output_type=AcceptanceCriteria,
        deps_type=ProjectDeps,
        system_prompt=BASE_SYSTEM_PROMPT,
    )

    @agent.system_prompt
    def inject_project_context(ctx: RunContext[ProjectDeps]) -> str:
        """
        This function runs at call time, reads from RunContext, and appends
        project-specific context to the system prompt.
        RunContext.deps holds the ProjectDeps object passed to agent.run().
        """
        return (
            f"Project: {ctx.deps.project_name}\n"
            f"Domain context: {ctx.deps.domain_context}"
        )

    return agent


# Ensemble: two agents, two models, one shared output contract.
agent_gpt    = create_agent("openai:gpt-4o")
agent_gemini = create_agent("google-gla:gemini-3-flash-preview")

ENSEMBLE = {
    "GPT-4o":        agent_gpt,
    "Gemini 3 Flash": agent_gemini,
}


# =============================================================================
# CONCEPT 5 — VOTING
#
# Each agent returns a fully validated AcceptanceCriteria object.
# The voting function receives a list of these and selects a winner.
#
# This spike uses criterion count as a naive proxy for quality:
# the output whose count matches the most agents wins. In production,
# replace this with semantic similarity scoring or an LLM-as-judge approach.
#
# Note: inputs are anonymised — the voting function receives data objects,
# not labelled results. It cannot tell which model produced which output.
# This prevents model-preference bias in scoring.
# =============================================================================

def majority_vote(
    outputs: list[AcceptanceCriteria],
) -> tuple[AcceptanceCriteria, str]:
    """
    Selects the AcceptanceCriteria object whose criterion count matches
    the most other agents (majority). Returns the winner and a brief rationale.
    Ties are broken in favour of the first-received result.
    """
    counts   = [len(ac.criteria) for ac in outputs]
    majority = Counter(counts).most_common(1)[0][0]
    winner   = next(ac for ac in outputs if len(ac.criteria) == majority)
    rationale = (
        f"Criterion counts per agent: {counts}. "
        f"Majority count: {majority}."
    )
    return winner, rationale


# =============================================================================
# CONCEPT 4 — PYTHON ORCHESTRATOR (asyncio.gather)
#
# run_ensemble() is the orchestrator. It:
#   1. Assembles the user prompt.
#   2. Dispatches to all agents in parallel via asyncio.gather.
#   3. Collects validated AcceptanceCriteria objects.
#   4. Passes them to the voting function.
#
# There is no framework here — this is plain asyncio.
# PydanticAI agents are async-native and work directly with gather.
# =============================================================================

async def run_ensemble(
    user_story: str,
    deps: ProjectDeps,
) -> AcceptanceCriteria:

    print("\n── Dispatching to ensemble (parallel) ──────────────────────────")

    # asyncio.gather runs all agent.run() calls concurrently.
    # Each returns a RunResult[AcceptanceCriteria]; .data holds the output.
    run_results = await asyncio.gather(
        *[agent.run(user_story, deps=deps) for agent in ENSEMBLE.values()]
    )

    outputs: list[AcceptanceCriteria] = [r.output for r in run_results]

    # Print each agent's output for inspection
    for label, output in zip(ENSEMBLE.keys(), outputs):
        print(f"\n  [{label}] — {len(output.criteria)} criterion/a")
        for i, c in enumerate(output.criteria, 1):
            print(f"    AC{i}  Given: {c.given}")
            print(f"         When:  {c.when}")
            print(f"         Then:  {c.then}")

    print("\n── Voting ───────────────────────────────────────────────────────")
    winner, rationale = majority_vote(outputs)
    print(f"  {rationale}")
    print(f"  Winner summary: '{winner.user_story_summary}'")

    return winner


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main() -> None:

    # Sample user story
    user_story = (
        "As a learner, I want to access the course dashboard when I select "
        "the dashboard tab, so I can view relevant course information and updates."
    )

    deps = ProjectDeps(
        project_name="UniLearn",
        domain_context=(
            "UniLearn is a university learning management system used by "
            "approximately 100,000 students. The course dashboard displays "
            "widgets including: Course Overview (enrolled courses and navigation), "
            "Progress Tracker (completed and pending tasks per learning mode: "
            "own-time, real-time, wrap-up), Community (discussions and "
            "announcements), and Support (learning resources and help)."
        ),
    )

    print("═" * 68)
    print("Project 30 — PydanticAI Spike")
    print("═" * 68)
    print(f"\nUser story:\n  {user_story}")

    winner = await run_ensemble(user_story, deps)

    print("\n── Final accepted output ────────────────────────────────────────")
    print(f"  Summary: {winner.user_story_summary}")
    for i, c in enumerate(winner.criteria, 1):
        print(f"\n  AC{i}")
        print(f"    Given: {c.given}")
        print(f"    When:  {c.when}")
        print(f"    Then:  {c.then}")

    print("\n── Key observations ─────────────────────────────────────────────")
    print("  • All three outputs were AcceptanceCriteria objects — PydanticAI")
    print("    enforced the schema. No manual parsing was required.")
    print("  • asyncio.gather ran all three agents concurrently, not serially.")
    print("  • The voting function received anonymous data — no model labels.")
    print("  • Replacing a model = changing one string in the ENSEMBLE dict.")
    print("═" * 68)


if __name__ == "__main__":
    asyncio.run(main())