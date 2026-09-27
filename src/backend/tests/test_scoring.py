import os

import pytest
from voting.models import EvaluatedOutput, RubricAverage, VotingResult

import app.services.scoring as scoring
from app.config import Settings
from app.services.scoring import CandidateScore, _scale, render_ac_candidate, score_candidates


def _rubric_averages(**values: float) -> list[RubricAverage]:
    return [RubricAverage(rubric=name, value=value) for name, value in values.items()]


def _evaluated_output(text: str, overall: float, **rubric_values: float) -> EvaluatedOutput:
    return EvaluatedOutput(
        output=text,
        feedback=[],
        rubric_averages=_rubric_averages(**rubric_values),
        overall_score=overall,
    )


def _voting_result(prompt: str, outputs: list[EvaluatedOutput]) -> VotingResult:
    return VotingResult(ai="test", model="test", prompt=prompt, output=outputs)


def test_scale_passes_voting_1_to_5_rubric_values_through_unchanged():
    assert _scale(1) == 1.0
    assert _scale(5) == 5.0
    assert _scale(3) == 3.0


def test_scale_clamps_the_error_sentinel_to_zero_rather_than_negative():
    assert _scale(-1) == 0.0


def test_score_candidates_short_circuits_on_empty_input(monkeypatch):
    called = False

    async def fail(_evaluation_input):
        nonlocal called
        called = True
        raise AssertionError("should not be called")

    monkeypatch.setattr(scoring, "_evaluate_input", fail)

    assert score_candidates("prompt", []) == []
    assert called is False


def test_score_candidates_matches_results_back_by_text_when_shuffled(monkeypatch):
    candidates = [
        render_ac_candidate("A", "ga", "wa", "ta"),
        render_ac_candidate("B", "gb", "wb", "tb"),
        render_ac_candidate("C", "gc", "wc", "tc"),
    ]

    async def fake_evaluate_input(evaluation_input):
        # Return results in reverse order and re-sorted by score, as the real
        # voting layer's _merge_provider_outputs does — score_candidates must
        # still return scores lined up with the ORIGINAL candidate order.
        outputs = [
            _evaluated_output(
                candidates[2], overall=5, relevance=5, correctness=5, understandability=5, coverage=5
            ),
            _evaluated_output(
                candidates[0], overall=1, relevance=1, correctness=1, understandability=1, coverage=1
            ),
            _evaluated_output(
                candidates[1], overall=3, relevance=3, correctness=3, understandability=3, coverage=3
            ),
        ]
        return _voting_result(evaluation_input.prompt, outputs)

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    scores = score_candidates("prompt", candidates)

    assert len(scores) == 3
    assert scores[0].overall == 1.0  # candidate A scored 1
    assert scores[1].overall == 3.0  # candidate B scored 3
    assert scores[2].overall == 5.0  # candidate C scored 5


def test_score_candidates_handles_duplicate_candidate_text(monkeypatch):
    duplicate = render_ac_candidate("Same", "g", "w", "t")
    candidates = [duplicate, duplicate]

    async def fake_evaluate_input(evaluation_input):
        outputs = [
            _evaluated_output(
                duplicate, overall=1, relevance=1, correctness=1, understandability=1, coverage=1
            ),
            _evaluated_output(
                duplicate, overall=5, relevance=5, correctness=5, understandability=5, coverage=5
            ),
        ]
        return _voting_result(evaluation_input.prompt, outputs)

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    scores = score_candidates("prompt", candidates)

    # Each occurrence consumes one queued result, in the order returned —
    # not the same result reused twice, and not skipped.
    assert scores[0].overall == 1.0
    assert scores[1].overall == 5.0


def test_score_candidates_falls_back_to_zero_when_the_provider_call_raises(monkeypatch):
    async def explode(_evaluation_input):
        raise RuntimeError("network is down")

    monkeypatch.setattr(scoring, "_evaluate_input", explode)

    scores = score_candidates("prompt", ["a", "b"])

    assert scores == [CandidateScore(0, 0, 0, 0, 0, failed=True), CandidateScore(0, 0, 0, 0, 0, failed=True)]


def test_score_candidates_falls_back_to_zero_when_results_dont_cover_all_candidates(monkeypatch):
    async def fake_evaluate_input(evaluation_input):
        # Only one result for two candidates submitted.
        return _voting_result(
            evaluation_input.prompt,
            [_evaluated_output("a", overall=5, relevance=5, correctness=5, understandability=5, coverage=5)],
        )

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    scores = score_candidates("prompt", ["a", "b"])

    assert scores == [CandidateScore(0, 0, 0, 0, 0, failed=True), CandidateScore(0, 0, 0, 0, 0, failed=True)]


def test_score_candidates_falls_back_to_zero_when_voting_failed_to_import(monkeypatch):
    monkeypatch.setattr(scoring, "_evaluate_input", None)

    scores = score_candidates("prompt", ["a"])

    assert scores == [CandidateScore(0, 0, 0, 0, 0, failed=True)]


def test_score_candidates_bridges_gemini_api_key_into_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    async def fake_evaluate_input(evaluation_input):
        return _voting_result(
            evaluation_input.prompt,
            [_evaluated_output("a", overall=5, relevance=5, correctness=5, understandability=5, coverage=5)],
        )

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    settings = Settings(gemini_api_key="secret-test-key")
    score_candidates("prompt", ["a"], settings=settings)

    assert os.environ.get("GEMINI_API_KEY") == "secret-test-key"


def test_score_candidates_does_not_overwrite_an_already_set_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "existing-key")

    async def fake_evaluate_input(evaluation_input):
        return _voting_result(
            evaluation_input.prompt,
            [_evaluated_output("a", overall=5, relevance=5, correctness=5, understandability=5, coverage=5)],
        )

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    settings = Settings(gemini_api_key="different-key")
    score_candidates("prompt", ["a"], settings=settings)

    assert os.environ.get("GEMINI_API_KEY") == "existing-key"


def test_score_marks_candidates_the_voting_layer_failed_to_rate(monkeypatch):
    """The -1 sentinel clamps to 0.0, so `failed` is the only way to tell."""
    async def fake_evaluate_input(evaluation_input):
        outputs = [
            _evaluated_output(
                "a", overall=-1, relevance=-1, correctness=-1,
                understandability=-1, coverage=-1,
            ),
            _evaluated_output(
                "b", overall=0, relevance=0, correctness=0,
                understandability=0, coverage=0,
            ),
        ]
        return _voting_result(evaluation_input.prompt, outputs)

    monkeypatch.setattr(scoring, "_evaluate_input", fake_evaluate_input)

    scores = scoring.score_candidates("prompt", ["a", "b"])

    assert scores[0].overall == 0.0 and scores[0].failed is True
    assert scores[1].overall == 0.0 and scores[1].failed is False


def test_zero_fallback_scores_are_marked_as_failed(monkeypatch):
    def boom(_):
        raise RuntimeError("voter down")

    monkeypatch.setattr(scoring, "_evaluate_input", boom)

    scores = scoring.score_candidates("prompt", ["a"])

    assert scores[0].failed is True
