"""The chat assistant, on stub models."""

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.models_conversation import (
    ConversationAttachment,
    ConversationMessage,
    ConversationRequest,
)
from app.services.conversation import ConversationService, render_conversation


def _request(*messages: ConversationMessage) -> ConversationRequest:
    """Wrap messages in a conversation request."""
    return ConversationRequest(messages=list(messages))


def _service(*, chat_model=None) -> ConversationService:
    """Build a service on an offline model, so no test needs an API key."""
    return ConversationService(
        chat_agent=Agent(
            chat_model if chat_model is not None else TestModel(),
            output_type=str,
            system_prompt="test",
        ),
    )


def test_attachments_are_inlined_as_plain_text_under_their_message():
    prompt = render_conversation(
        _request(
            ConversationMessage(
                role="user",
                text="Here is the section menu.",
                attachments=[
                    ConversationAttachment(
                        filename="001-1.png",
                        content="Mass Actions\nSelect all in section",
                    )
                ],
            )
        )
    )

    assert "USER: Here is the section menu." in prompt
    assert "--- attached file: 001-1.png ---" in prompt
    assert "Select all in section" in prompt
    assert "--- end of 001-1.png ---" in prompt


def test_the_whole_conversation_is_rendered_in_order():
    prompt = render_conversation(
        _request(
            ConversationMessage(role="user", text="First story"),
            ConversationMessage(role="assistant", text="Understood"),
            ConversationMessage(role="user", text="Second story"),
        )
    )

    assert prompt.index("First story") < prompt.index("Understood")
    assert prompt.index("Understood") < prompt.index("Second story")
    assert "ASSISTANT: Understood" in prompt


def test_send_message_returns_the_models_reply():
    def echo(messages, agent_info):
        return ModelResponse(parts=[TextPart(content="Real assistant reply")])

    service = _service(chat_model=FunctionModel(echo))

    reply = service.send_message(_request(ConversationMessage(role="user", text="hi")))

    assert reply.role == "assistant"
    assert reply.text == "Real assistant reply"


def test_send_message_falls_back_to_canned_reply_on_model_failure():
    def explode(messages, agent_info):
        raise RuntimeError("upstream is down")

    service = _service(chat_model=FunctionModel(explode))

    reply = service.send_message(_request(ConversationMessage(role="user", text="hi")))

    assert reply.role == "assistant"
    assert reply.text  # The canned reply.


def test_the_chat_model_receives_the_attachment_text():
    prompts_seen: list[str] = []

    def respond(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        prompt_part = messages[-1].parts[-1]
        assert isinstance(prompt_part, UserPromptPart)
        prompts_seen.append(str(prompt_part.content))
        return ModelResponse(parts=[TextPart(content="Thanks, that helps.")])

    service = _service(chat_model=FunctionModel(respond))

    service.send_message(
        _request(
            ConversationMessage(
                role="user",
                text="Read this",
                attachments=[
                    ConversationAttachment(
                        filename="spec.pdf", content="Sections may be reordered."
                    )
                ],
            )
        )
    )

    assert "Sections may be reordered." in prompts_seen[0]
    assert "spec.pdf" in prompts_seen[0]
