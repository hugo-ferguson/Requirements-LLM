from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from app.config import Settings, settings as default_settings
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
    ConversationRequest,
    GenerateResult,
)

logger = logging.getLogger(__name__)

_CANNED_ASSISTANT_REPLY = (
    "From viewing the user stories there seems to be enough context, "
    "feel free to generate the acceptance criteria."
)

SYSTEM_PROMPT = """\
You are an expert business analyst writing acceptance criteria for software features.
Given a conversation containing user stories and supporting material, produce clear,
atomic, testable Gherkin acceptance criteria in Given / When / Then format.

Each criterion should cover one distinct user-observable behaviour. Avoid
implementation detail. Avoid vague language like "the system works correctly".

Some messages carry attached files. Their contents are machine transcriptions —
text pulled out of screenshots by a vision model, or extracted from a PDF — so
the wording is reliable but the layout is not. Treat labels, menu items and
field names in them as evidence of what the interface actually offers, and say
so plainly rather than speculating about anything the transcript does not show.
"""


class GenerationError(RuntimeError):
    """Raised when the model could not be reached or returned nothing usable."""


class EmptyConversationError(ValueError):
    """
    Raised when there is nothing to generate acceptance criteria from.

    Worth guarding explicitly: asked to work from an empty conversation, the
    model does not refuse — it invents a plausible feature and writes criteria
    for it, which is far worse than an error.
    """


class GeneratedCriterion(BaseModel):
    """One acceptance criterion, as written by the model."""

    title: str = Field(
        description="Short label for the behaviour, e.g. 'Successful login → /home'"
    )
    given: str = Field(description="Precondition or initial system/user state")
    when: str = Field(description="The specific action or event being tested")
    then: str = Field(description="The expected, observable outcome")


class GeneratedCriteria(BaseModel):
    criteria: list[GeneratedCriterion] = Field(
        description="Gherkin-style acceptance criteria, one per testable behaviour",
        min_length=1,
    )


def build_agent(settings: Settings) -> Agent[None, GeneratedCriteria]:
    """
    Returns the acceptance-criteria agent for the configured model.

    The API key is passed explicitly rather than left to the ambient
    environment, so the app has one place that decides where credentials come
    from.
    """
    provider, _, model_name = settings.llm_model.partition(":")

    if provider == "google":
        model = GoogleModel(
            model_name,  # type: ignore[arg-type]
            provider=GoogleProvider(api_key=settings.gemini_api_key),
        )
    else:
        # Any other PydanticAI model string still works, but it has to find
        # its own credentials in the environment.
        model = settings.llm_model  # type: ignore[assignment]

    return Agent(model, output_type=GeneratedCriteria, system_prompt=SYSTEM_PROMPT)


def render_conversation(request: ConversationRequest) -> str:
    """
    Flattens the conversation into the plain text handed to the model.

    Attachments are inlined as plain transcripts under the message that
    carried them, so the model reads a screenshot's text in the context of
    whatever the user said about it.
    """
    return "\n\n".join(_render_message(message) for message in request.messages)


def _has_content(request: ConversationRequest) -> bool:
    """True when any message carries text or an attachment worth reading."""
    return any(
        message.text.strip()
        or any(a.content.strip() for a in message.attachments)
        for message in request.messages
    )


def _render_message(message: ConversationMessage) -> str:
    parts = [f"{message.role.upper()}: {message.text}".rstrip()]

    for attachment in message.attachments:
        parts.append(
            f"--- attached file: {attachment.filename} ---\n"
            f"{attachment.content.strip()}\n"
            f"--- end of {attachment.filename} ---"
        )

    return "\n".join(parts)


class ConversationService:
    """
    Conversation logic, independent of HTTP concerns.

    `generate` is wired to a real model; `send_message` is still a stub that
    returns a fixed reply.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        agent: Agent[None, GeneratedCriteria] | None = None,
    ):
        self.settings = settings or default_settings
        # Built on first use so that a missing API key breaks generation
        # rather than every request that happens to construct this service.
        self._agent = agent

    @property
    def agent(self) -> Agent[None, GeneratedCriteria]:
        if self._agent is None:
            self._agent = build_agent(self.settings)
        return self._agent

    def send_message(self, data: ConversationRequest) -> ConversationMessage:
        return ConversationMessage(role="assistant", text=_CANNED_ASSISTANT_REPLY)

    def generate(self, data: ConversationRequest) -> GenerateResult:
        if not _has_content(data):
            raise EmptyConversationError(
                "Add a user story or attach a document before generating "
                "acceptance criteria."
            )

        prompt = render_conversation(data)
        logger.info(
            "Generating acceptance criteria from %s message(s), %s characters",
            len(data.messages),
            len(prompt),
        )

        try:
            result = self.agent.run_sync(prompt)
        except Exception as error:
            raise GenerationError(
                f"{self.settings.llm_model} could not generate acceptance "
                f"criteria: {error}"
            ) from error

        return GenerateResult(
            acceptance_criteria=[
                AcceptanceCriterion(
                    # Positional only — the repository assigns real ids when
                    # it persists the batch.
                    id=index,
                    title=criterion.title,
                    given=criterion.given,
                    when=criterion.when,
                    then=criterion.then,
                    # Scoring belongs to the voting layer, which isn't wired
                    # in yet. Left at zero rather than invented here.
                    scores=AcceptanceCriterionScores(
                        relevance=0, correctness=0, understandability=0, coverage=0
                    ),
                    overall_score=0,
                )
                for index, criterion in enumerate(result.output.criteria, start=1)
            ]
        )
