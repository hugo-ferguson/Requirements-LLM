from voting.models import EvaluationInput, JudgeConfig


def test_judges_travel_on_the_evaluation_input() -> None:
    judge = JudgeConfig(id="gemini", model="gemini/gemini-3.6-flash")
    evaluation = EvaluationInput(
        ai="test-ai",
        model="test-model",
        prompt="Check this response",
        output=["A generated answer"],
        judges=[judge],
    )

    assert evaluation.judges == [judge]


def test_judge_defaults_send_nothing_a_provider_might_reject() -> None:
    judge = JudgeConfig(id="claude", model="anthropic/claude-sonnet-5-5")

    assert judge.temperature is None
    assert judge.cache_prompt is False
    assert judge.style == "combined"
    assert judge.display_name == "Claude"
