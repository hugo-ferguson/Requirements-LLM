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
