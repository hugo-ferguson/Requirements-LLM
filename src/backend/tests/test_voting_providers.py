from voting.gemini import _parse_response_text
from voting.models import EvaluationInput


def test_gemini_is_a_valid_provider() -> None:
    evaluation = EvaluationInput(
        ai="test-ai",
        model="test-model",
        prompt="Check this response",
        output=["A generated answer"],
        providers=["gemini"],
    )

    assert evaluation.providers == ["gemini"]


def test_parse_response_text_normalizes_capitalized_gemini_json() -> None:
    raw = '{"Correctness": {"Feedback": "Looks good", "Score": 4}, "Coverage": {"Feedback": "Mostly covered", "Score": 3}, "Relevance": {"Feedback": "Relevant", "Score": 4}, "Understandability": {"Feedback": "Clear", "Score": 5}}'

    parsed = _parse_response_text(raw)

    assert parsed.correctness.score == 4
    assert parsed.coverage.score == 3
    assert parsed.relevance.feedback == "Relevant"
    assert parsed.understandability.score == 5
