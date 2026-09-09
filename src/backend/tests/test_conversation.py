import pytest
from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelResponse, TextPart

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
from app.services.scoring import CandidateScore


def _request(*messages: ConversationMessage) -> ConversationRequest:
    return ConversationRequest(messages=list(messages))


def _zero_scorer(prompt, candidates, *, reference_answer="", settings=None):
    """A fake scorer standing in for the voting layer, so generate-only tests
    stay offline and don't depend on the real scoring wiring under test."""
    return [CandidateScore(0, 0, 0, 0, 0) for _ in candidates]


def _service(model, *, chat_model=None, scorer=None) -> ConversationService:
    """A service backed by offline models, so no test needs an API key."""
    return ConversationService(
        agent=Agent(model, output_type=GeneratedCriteria, system_prompt="test"),
        chat_agent=(
            Agent(chat_model, output_type=str, system_prompt="test")
            if chat_model is not None
            else Agent(TestModel(), output_type=str, system_prompt="test")
        ),
        scorer=scorer or _zero_scorer,
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


def test_generate_scores_criteria_via_the_injected_scorer():
    seen_prompts_and_candidates: list[tuple[str, list[str]]] = []

    def fake_scorer(prompt, candidates, *, reference_answer="", settings=None):
        seen_prompts_and_candidates.append((prompt, list(candidates)))
        return [
            CandidateScore(relevance=1, correctness=2, understandability=3, coverage=4, overall=5)
            for _ in candidates
        ]

    service = _service(TestModel(), scorer=fake_scorer)

    result = service.generate(_request(ConversationMessage(role="user", text="story")))

    assert result.acceptance_criteria
    for criterion in result.acceptance_criteria:
        assert criterion.scores.relevance == 1
        assert criterion.scores.correctness == 2
        assert criterion.scores.understandability == 3
        assert criterion.scores.coverage == 4
        assert criterion.overall_score == 5

    # The scorer is called with the same rendered prompt used for
    # generation, and one candidate string per generated criterion.
    prompt, candidates = seen_prompts_and_candidates[0]
    assert prompt == render_conversation(_request(ConversationMessage(role="user", text="story")))
    assert len(candidates) == len(result.acceptance_criteria)


def test_send_message_returns_the_models_reply():
    def echo(messages, info):
        return ModelResponse(parts=[TextPart(content="Real assistant reply")])

    service = _service(TestModel(), chat_model=FunctionModel(echo))

    reply = service.send_message(_request(ConversationMessage(role="user", text="hi")))

    assert reply.role == "assistant"
    assert reply.text == "Real assistant reply"


def test_send_message_falls_back_to_canned_reply_on_model_failure():
    def explode(messages, info):
        raise RuntimeError("upstream is down")

    service = _service(TestModel(), chat_model=FunctionModel(explode))

    reply = service.send_message(_request(ConversationMessage(role="user", text="hi")))

    assert reply.role == "assistant"
    assert reply.text  # falls back to the canned string, never raises


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
