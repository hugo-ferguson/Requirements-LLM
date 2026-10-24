"""The chat assistant that helps users flesh out their user stories."""

from __future__ import annotations

import logging

from pydantic_ai import Agent

from app.models_conversation import ConversationMessage, ConversationRequest
from app.services.agents import build_agent

logger = logging.getLogger(__name__)

# Sent when the chat model call fails.
_CANNED_ASSISTANT_REPLY = (
    "From viewing the user stories there seems to be enough context, "
    "feel free to generate the acceptance criteria."
)

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


def build_chat_agent() -> Agent[None, str]:
    """Return the chat-reply agent for the configured chat model."""
    return build_agent(str, CHAT_SYSTEM_PROMPT)


def render_conversation(request: ConversationRequest) -> str:
    """Flatten the conversation into plain text for the model.

    Each attachment's text goes under the message that carried it.
    """
    return "\n\n".join(_render_message(message) for message in request.messages)


def _render_message(message: ConversationMessage) -> str:
    """Render one "ROLE: text" line, then each attachment fenced off."""
    parts = [f"{message.role.upper()}: {message.text}".rstrip()]

    for attachment in message.attachments:
        parts.append(
            f"--- attached file: {attachment.filename} ---\n"
            f"{attachment.content.strip()}\n"
            f"--- end of {attachment.filename} ---"
        )

    return "\n".join(parts)


class ConversationService:
    """Chat replies for a session, independent of HTTP concerns."""

    def __init__(self, chat_agent: Agent[None, str] | None = None):
        """Store an optional prebuilt chat agent, for tests."""
        # Built lazily, so a missing API key gives the canned reply.
        self._chat_agent = chat_agent

    @property
    def chat_agent(self) -> Agent[None, str]:
        """The chat agent, built on first use."""
        if self._chat_agent is None:
            self._chat_agent = build_chat_agent()
        return self._chat_agent

    def send_message(self, request: ConversationRequest) -> ConversationMessage:
        """Return the assistant's reply, or a canned reply if the model fails."""
        prompt = render_conversation(request)
        try:
            result = self.chat_agent.run_sync(prompt)
            text = result.output
        except Exception:
            logger.warning(
                "Chat agent call failed; falling back to canned reply", exc_info=True
            )
            text = _CANNED_ASSISTANT_REPLY
        return ConversationMessage(role="assistant", text=text)
