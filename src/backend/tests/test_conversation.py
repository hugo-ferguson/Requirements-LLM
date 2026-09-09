import pytest
from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelResponse

from app.models_conversation import (
    ConversationAttachment,
    ConversationMessage,
    ConversationRequest,
)
from app.services.conversation import (
    EmptyConversationError,
    GeneratedCriteria,
    GenerationError,
    ConversationService,
    render_conversation,
)


def _request(*messages: ConversationMessage) -> ConversationRequest:
    return ConversationRequest(messages=list(messages))


def _service(model) -> ConversationService:
    """A service backed by an offline model, so no test needs an API key."""
    return ConversationService(
        agent=Agent(model, output_type=GeneratedCriteria, system_prompt="test")
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


def test_generate_returns_criteria_from_the_model():
    service = _service(TestModel())

    result = service.generate(_request(ConversationMessage(role="user", text="story")))

    assert result.acceptance_criteria
    first = result.acceptance_criteria[0]
    assert first.title and first.given and first.when and first.then
    # Ids are positional; the repository assigns the real ones on persist.
    assert [c.id for c in result.acceptance_criteria] == list(
        range(1, len(result.acceptance_criteria) + 1)
    )


def test_generated_criteria_are_left_unscored_for_the_voting_layer():
    service = _service(TestModel())

    result = service.generate(_request(ConversationMessage(role="user", text="story")))

    for criterion in result.acceptance_criteria:
        assert criterion.overall_score == 0
        assert criterion.scores.relevance == 0
        assert criterion.scores.coverage == 0


def test_the_model_actually_receives_the_attachment_text():
    seen: list[str] = []

    service = _service(
        FunctionModel(
            lambda messages, info: (
                seen.append(str(messages[-1].parts[-1].content)),  # type: ignore[union-attr]
                _tool_response(info),
            )[1]
        )
    )

    service.generate(
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

    assert "Sections may be reordered." in seen[0]
    assert "spec.pdf" in seen[0]


def _tool_response(info: AgentInfo) -> ModelResponse:
    """Answers with the structured output the agent asked for."""
    from pydantic_ai.messages import ToolCallPart

    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={
                    "criteria": [
                        {
                            "title": "Reorder a section",
                            "given": "a unit with several sections",
                            "when": "an administrator moves one before another",
                            "then": "the new order is shown and saved",
                        }
                    ]
                },
            )
        ]
    )


def test_a_failing_model_raises_generation_error():
    def explode(messages, info):
        raise RuntimeError("upstream is down")

    service = _service(FunctionModel(explode))

    with pytest.raises(GenerationError) as caught:
        service.generate(_request(ConversationMessage(role="user", text="story")))

    assert "upstream is down" in str(caught.value)


def test_generating_from_an_empty_conversation_is_refused_without_calling_the_model():
    """
    Guards a real failure seen in testing: given nothing to work from, the
    model happily invented an unrelated feature and wrote criteria for it.
    """
    def explode(messages, info):  # must never be reached
        raise AssertionError("the model should not have been called")

    service = _service(FunctionModel(explode))

    for request in (
        _request(),
        _request(ConversationMessage(role="user", text="   ")),
        _request(
            ConversationMessage(
                role="user",
                text="",
                attachments=[ConversationAttachment(filename="blank.txt", content=" ")],
            )
        ),
    ):
        with pytest.raises(EmptyConversationError):
            service.generate(request)
