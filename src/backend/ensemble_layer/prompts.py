"""
Prompt templates for the Generation Agents.

The output schema is enforced by PydanticAI via `output_type` (see agent.py),
so this file only carries role/task/quality guidance — not formatting rules.
"""

SYSTEM_PROMPT = """\
You are an expert QA engineer and business analyst specialising in \
writing Gherkin-format acceptance criteria for software features.

Your job is to analyse a user story and produce a set of clear, \
testable acceptance criteria in Gherkin (Given/When/Then) format.

Guidelines:
- Write 4 to 6 scenarios per user story.
- Cover the happy path AND key edge/error cases.
- Steps should be concrete and specific — avoid vague language like "the system works".
"""


def build_user_prompt(
    story_title: str,
    story_description: str,
    additional_context: str | None = None,
    ui_context: str | None = None,
) -> str:
    """Assemble the user-turn prompt from story fields."""
    parts = [
        f"## User Story\n**{story_title}**\n\n{story_description}",
    ]
    if additional_context:
        parts.append(f"## Backend / Domain Context\n{additional_context}")
    if ui_context:
        parts.append(f"## UI Context (from mockup)\n{ui_context}")
    return "\n\n".join(parts)
