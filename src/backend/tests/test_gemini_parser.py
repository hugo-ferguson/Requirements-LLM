from voting.gemini import _parse_response_text


def test_parse_response_text_handles_string_rubric_values() -> None:
    payload = {
        "correctness": "score: 4/5\nfeedback: The response is accurate and internally consistent.",
        "coverage": "score: 3/5\nfeedback: It covers the main path but misses a few edge cases.",
        "relevance": "score: 5/5\nfeedback: It stays focused on the requested requirement.",
        "understandability": "score: 4/5\nfeedback: The wording is clear and easy to follow.",
    }

    vote = _parse_response_text(str(payload).replace("'", '"'))

    assert vote.correctness.score == 4
    assert vote.coverage.score == 3
    assert vote.relevance.score == 5
    assert vote.understandability.score == 4
    assert "accurate" in vote.correctness.feedback.lower()


def test_parse_response_text_handles_unstructured_gemini_text() -> None:
    raw = """The response is a single declarative statement rather than a proper set of acceptance criteria.
    correctness: the response lacks testable criteria.
    coverage: it misses the main workflow.
    relevance: it is partly on topic but not enough.
    understandability: it is clear but not structured enough."""

    vote = _parse_response_text(raw)

    assert vote.correctness.score == -1
    assert vote.coverage.score == -1
    assert vote.relevance.score == -1
    assert vote.understandability.score == -1
    assert "correctness" in vote.correctness.feedback.lower() or "response" in vote.correctness.feedback.lower()


def test_gemini_request_forces_json_schema() -> None:
    from voting.gemini import _build_generation_config

    config = _build_generation_config()

    assert config["responseMimeType"] == "application/json"
    assert "responseSchema" in config
    assert config["responseSchema"]["type"] == "OBJECT"
    assert config["responseSchema"]["required"] == [
        "correctness",
        "coverage",
        "relevance",
        "understandability",
    ]


def test_evaluated_outputs_are_sorted_by_overall_score_desc() -> None:
    from voting.models import EvaluationInput, EvaluatedOutput, ProviderFeedback, RubricAverage, RubricFeedback
    from voting.voting import _merge_provider_outputs

    evaluation_input = EvaluationInput(
        ai="sample",
        model="sample-model",
        prompt="test prompt",
        output=["first output", "second output"],
        providers=["gemini"],
    )

    provider_outputs = [[
        EvaluatedOutput(
            output="first output",
            feedback=[
                ProviderFeedback(
                    ai="Gemini",
                    model="gemini",
                    feedback=[RubricFeedback(rubric="correctness", value=3, feedback="ok")],
                    overall_score=3.0,
                )
            ],
            rubric_averages=[RubricAverage(rubric="correctness", value=3.0)],
            overall_score=3.0,
        ),
        EvaluatedOutput(
            output="second output",
            feedback=[
                ProviderFeedback(
                    ai="Gemini",
                    model="gemini",
                    feedback=[RubricFeedback(rubric="correctness", value=5, feedback="great")],
                    overall_score=5.0,
                )
            ],
            rubric_averages=[RubricAverage(rubric="correctness", value=5.0)],
            overall_score=5.0,
        ),
    ]]

    result = _merge_provider_outputs(evaluation_input, provider_outputs)

    assert [item.output for item in result.output] == ["second output", "first output"]
    assert result.output[0].overall_score >= result.output[1].overall_score
