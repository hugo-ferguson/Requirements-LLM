from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.config import Settings, settings as default_settings
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
    ConversationRequest,
    GenerateResult,
)
from app.services.agents import GenerationError, build_agent as _build_shared_agent
from app.services.scoring import Scorer, render_ac_candidate, score_candidates

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationService",
    "EmptyConversationError",
    "GenerationError",
    "GeneratedCriteria",
    "GeneratedCriterion",
    "build_agent",
    "build_chat_agent",
    "render_conversation",
]

# Fallback only — used when the real chat agent call fails, so chat never
# hard-crashes the conversation.
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

CHAT_SYSTEM_PROMPT = """\
You are a friendly, sharp business-analyst assistant helping a teammate flesh
out user stories before acceptance criteria get generated from them.

Read the conversation so far — user messages, your own previous replies, and
any attached file transcripts — and reply with ONE short, conversational
message. Ask a clarifying question if something material is missing or
ambiguous (a specific field, error state, permission rule, edge case).
Otherwise tell them plainly there's enough context and they're welcome to
generate acceptance criteria now. Never write acceptance criteria yourself
here — that only happens through the dedicated Generate step.
"""


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
    """Returns the acceptance-criteria agent for the configured model."""
    return _build_shared_agent(settings, GeneratedCriteria, SYSTEM_PROMPT)


def build_chat_agent(settings: Settings) -> Agent[None, str]:
    """Returns the conversational chat-reply agent for the configured model."""
    return _build_shared_agent(settings, str, CHAT_SYSTEM_PROMPT)


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

    Both `generate` and `send_message` are wired to real models; `generate`
    also scores its output via the voting layer (through `scorer`).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        agent: Agent[None, GeneratedCriteria] | None = None,
        chat_agent: Agent[None, str] | None = None,
        scorer: Scorer = score_candidates,
    ):
        self.settings = settings or default_settings
        # Built on first use so that a missing API key breaks generation
        # rather than every request that happens to construct this service.
        self._agent = agent
        self._chat_agent = chat_agent
        self.scorer = scorer

    @property
    def agent(self) -> Agent[None, GeneratedCriteria]:
        if self._agent is None:
            self._agent = build_agent(self.settings)
        return self._agent

    @property
    def chat_agent(self) -> Agent[None, str]:
        if self._chat_agent is None:
            self._chat_agent = build_chat_agent(self.settings)
        return self._chat_agent

    def send_message(self, data: ConversationRequest) -> ConversationMessage:
        prompt = render_conversation(data)
        try:
            result = self.chat_agent.run_sync(prompt)
            text = result.output
        except Exception:
            # Chat is conversational, not the pipeline's critical path — a
            # failed call falls back to a generic reply rather than
            # breaking the conversation.
            logger.warning("Chat agent call failed; falling back to canned reply", exc_info=True)
            text = _CANNED_ASSISTANT_REPLY
        return ConversationMessage(role="assistant", text=text)

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

        criteria = result.output.criteria
        candidate_texts = [
            render_ac_candidate(c.title, c.given, c.when, c.then) for c in criteria
        ]
        scores = self.scorer(prompt=prompt, candidates=candidate_texts, settings=self.settings)

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
                    scores=AcceptanceCriterionScores(
                        relevance=score.relevance,
                        correctness=score.correctness,
                        understandability=score.understandability,
                        coverage=score.coverage,
                    ),
                    overall_score=score.overall,
                )
                for index, (criterion, score) in enumerate(
                    zip(criteria, scores, strict=True), start=1
                )
            ]
        )
