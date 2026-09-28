"""Prompt construction for the generation agents.

The system prompt is identical across every agent in the ensemble — the only
thing that varies is the model behind it. That is what makes the outputs
comparable when the voting layer scores them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


SYSTEM_PROMPT = """\
You are an experienced business analyst writing acceptance criteria for a \
software feature.

Given a user story and any supporting project context, produce clear, atomic, \
testable acceptance criteria in Gherkin Given / When / Then form.

Rules:
- Each criterion covers exactly one distinct, user-observable behaviour.
- Cover the happy path first, then negative paths, boundaries and error states.
- Write what the system does, never how it is implemented. No database tables, \
API endpoints, class names or frameworks.
- Never use vague wording such as "works correctly", "as expected", or \
"appropriately".
- Each of given / when / then is a single sentence fragment written WITHOUT the \
leading keyword. Write "the user is on the login page", not "Given the user is \
on the login page".
- The title is a short imperative label of at most 8 words.
- Only use facts present in the user story or the supplied project context. If \
something is genuinely unspecified, choose the most conventional behaviour \
rather than inventing product-specific detail.
"""


@dataclass
class GenerationDeps:
    """Runtime dependencies injected into an agent run via PydanticAI's
    `RunContext`.

    Holding these in an object rather than string-formatting them into the user
    prompt means the same agent instance serves every session — nothing about a
    particular request is baked into the agent at construction time.
    """

    user_story: str
    context_chunks: list[str] = field(default_factory=list)
    feedback: str | None = None

    def context_block(self) -> str:
        """Render retrieved RAG chunks as an appended system message."""
        if not self.context_chunks:
            return (
                "No project context documents were retrieved for this story. "
                "Base the criteria on the user story alone."
            )

        numbered = "\n\n".join(
            f"[Context {index}]\n{chunk.strip()}"
            for index, chunk in enumerate(self.context_chunks, start=1)
        )
        return (
            "The following excerpts were retrieved from the project's own "
            "documents. Prefer terminology and behaviour described here over "
            "generic assumptions.\n\n" + numbered
        )

    def feedback_block(self) -> str | None:
        """Render reviewer feedback for a regeneration pass (backlog R20)."""
        if not self.feedback:
            return None
        return (
            "A reviewer rejected a previous attempt and gave this feedback. "
            "Address it directly in this generation:\n\n" + self.feedback.strip()
        )


def build_user_prompt(deps: GenerationDeps, max_criteria: int) -> str:
    """The per-run user message. Context and feedback go in via system prompts,
    so this stays short and stable."""
    return (
        f"Write up to {max_criteria} acceptance criteria for the following "
        f"user story.\n\nUser story:\n{deps.user_story.strip()}"
    )


def build_titles_prompt(deps: GenerationDeps, max_criteria: int) -> str:
    """Pass 1 of title-anchored generation: one agent fixes the title set.

    The titles become the grouping key every other agent writes against, which
    is what makes candidates from different models comparable. Descriptions are
    requested here too — this agent's own criteria are a full candidate in
    their own right, so asking for titles alone would waste the call.
    """
    return (
        f"Write up to {max_criteria} acceptance criteria for the following "
        f"user story. Each title must name a distinct behaviour — two titles "
        f"covering the same behaviour is the one thing to avoid here, because "
        f"these titles are the definitive list every reviewer will work "
        f"from.\n\nUser story:\n{deps.user_story.strip()}"
    )


def build_descriptions_prompt(
    deps: GenerationDeps, titles: list[str]
) -> str:
    """Pass 2: write given/when/then for an already-fixed set of titles.

    Every agent answers the same titles, so their outputs line up one-to-one
    and the voting layer compares like against like instead of scoring an
    arbitrary union. Answers are identified by title number, not by copying
    the title back: the number is the join key, and the orchestrator attaches
    the anchor's exact title itself.
    """
    numbered = "\n".join(f"{index}. {title}" for index, title in enumerate(titles, start=1))
    return (
        "Below is a fixed, numbered list of acceptance criteria titles for a user "
        "story. Write your own given / when / then for EVERY title in the list.\n\n"
        "Rules for this task:\n"
        "- Return exactly one entry per title, identified by its title_number "
        "from the list.\n"
        "- Do not skip, merge or add titles.\n"
        "- Write the strongest given / when / then you can for the behaviour "
        "the title names.\n\n"
        f"Titles:\n{numbered}\n\n"
        f"User story:\n{deps.user_story.strip()}"
    )


UAT_SYSTEM_PROMPT = """\
You are a QA engineer writing User Acceptance Test cases for one accepted
Gherkin acceptance criterion (Given/When/Then).

Each test case needs a short title and a description with concrete steps/data
a human tester could execute and the expected result. Stay grounded in what
the criterion actually states — do not invent unrelated functionality.
"""


def build_uat_cases_prompt(acceptance_criterion: str) -> str:
    """Pass 1 for UAT: the anchor agent writes the test cases for one AC."""
    return (
        "Write 2-4 concrete UAT test cases for the acceptance criterion below. "
        "Cover the happy path and at least one edge or negative case. Each "
        "title must name a distinct scenario — these titles are the list every "
        "other reviewer will work from.\n\n"
        f"Acceptance criterion:\n{acceptance_criterion.strip()}"
    )


def build_uat_descriptions_prompt(acceptance_criterion: str, titles: list[str]) -> str:
    """Pass 2 for UAT: write a description for each fixed, numbered case title.

    Answers are identified by number, like `build_descriptions_prompt`, so a
    reworded title can't drop an agent's answer.
    """
    numbered = "\n".join(f"{index}. {title}" for index, title in enumerate(titles, start=1))
    return (
        "Below is a fixed, numbered list of UAT test case titles for an "
        "acceptance criterion. Write your own description for EVERY title.\n\n"
        "Rules for this task:\n"
        "- Return exactly one entry per title, identified by its title_number "
        "from the list.\n"
        "- Do not skip, merge or add test cases.\n"
        "- Give concrete steps/data and the expected result.\n\n"
        f"Test case titles:\n{numbered}\n\n"
        f"Acceptance criterion:\n{acceptance_criterion.strip()}"
    )
