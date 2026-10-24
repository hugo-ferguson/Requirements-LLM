"""Score generated candidates through the voting layer.

Failures come back as all-zero scores marked `failed`, never as errors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.config import Settings
from app.config import settings as default_settings
from app.llm_config import get_models_config

if TYPE_CHECKING:
    from voting.models import EvaluatedOutput, JudgeConfig

logger = logging.getLogger(__name__)

# If the voting package fails to import, scoring falls back to zeros.
try:
    from voting.models import EvaluationInput
    from voting.voting import evaluate_input as _evaluate_input
except Exception as _import_error:  # pragma: no cover
    EvaluationInput = None  # type: ignore[assignment,misc]
    _evaluate_input = None
    _VOTING_IMPORT_ERROR: Exception | None = _import_error
else:
    _VOTING_IMPORT_ERROR = None

# The app's score range. Voting scores 1-5, so only the -1 error value moves.
_APP_MIN, _APP_MAX = 0.0, 5.0


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's 0-5 rubric scores."""

    relevance: float
    correctness: float
    understandability: float
    coverage: float
    overall: float
    # Never scored, so the zeros are placeholders. Ranked last.
    failed: bool = False


_ZERO_SCORE = CandidateScore(0.0, 0.0, 0.0, 0.0, 0.0, failed=True)


# One shared loop, so each judge's `max_parallel` limit covers concurrent runs.
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _voting_loop() -> asyncio.AbstractEventLoop:
    """Return the shared voting loop, starting it on first use."""
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever, name="voting-loop", daemon=True
            ).start()
        return _loop


class Scorer(Protocol):
    """Anything shaped like `score_candidates`, so tests can swap it."""

    def __call__(
        self,
        prompt: str,
        candidates: Sequence[str],
        *,
        reference_answer: str = "",
        settings: Settings | None = None,
    ) -> list[CandidateScore]:
        """Return one score per candidate, in the order given."""
        ...


def render_ac_candidate(title: str, given: str, when: str, then: str) -> str:
    """Render one acceptance criterion as flat text.

    Keep this the only definition, as scores are matched back by exact text.
    """
    return f"Title: {title}\nGiven: {given}\nWhen: {when}\nThen: {then}"


def render_uat_candidate(title: str, description: str) -> str:
    """Render one UAT test case; see `render_ac_candidate`."""
    return f"Title: {title}\nDescription: {description}"


def score_candidates(
    prompt: str,
    candidates: Sequence[str],
    *,
    reference_answer: str = "",
    settings: Settings | None = None,
    judges: Sequence[JudgeConfig] | None = None,
) -> list[CandidateScore]:
    """Score each candidate against `prompt` through the voting layer.

    Never raises. On failure every candidate gets a zero score marked `failed`.

    Args:
        prompt: The prompt the candidates were generated from.
        candidates: Rendered candidate texts, in display order.
        reference_answer: Passed to the voting layer, which ignores it.
        settings: App settings, for the Gemini API key.
        judges: The judges to use. Defaults to the enabled judges.

    Returns:
        One score per candidate, in the same order as `candidates`.
    """
    if not candidates:
        return []

    if _evaluate_input is None or EvaluationInput is None:
        logger.warning(
            "voting layer unavailable, scoring %d candidate(s) as zero: %s",
            len(candidates),
            _VOTING_IMPORT_ERROR,
        )
        return [_ZERO_SCORE] * len(candidates)

    resolved_settings = settings or default_settings
    if resolved_settings.gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
        # Gemini judges read os.environ, which doesn't include .env values.
        os.environ["GEMINI_API_KEY"] = resolved_settings.gemini_api_key

    try:
        selected_judges = (
            list(judges) if judges is not None else get_models_config().enabled_judges()
        )
        evaluation_input = EvaluationInput(
            ai="requirements-llm",
            model="+".join(judge.id for judge in selected_judges),
            prompt=prompt,
            output=list(candidates),
            reference_answer=reference_answer,
            judges=selected_judges,
        )
        voting_result = asyncio.run_coroutine_threadsafe(
            _evaluate_input(evaluation_input), _voting_loop()
        ).result()
        scores = _match_to_candidates(candidates, voting_result.output)

        unrated = sum(1 for score in scores if score.failed)
        if unrated and unrated != len(scores):
            logger.warning(
                "%d of %d candidate(s) were not rated by the voting layer and "
                "scored 0.0 by default; they will lose to any rated candidate "
                "regardless of quality",
                unrated,
                len(scores),
            )

        if scores and all(score.failed for score in scores):
            # A whole batch unscored usually means a judge is misconfigured.
            logger.warning(
                "every one of %d candidate(s) scored 0.0 — this is almost always a "
                "misconfigured voter rather than genuinely worthless criteria. "
                "Check the judges in config/models.json (%s) and that their models "
                "are reachable; see the judge warning logged above for the cause.",
                len(scores),
                ", ".join(repr(judge.id) for judge in selected_judges),
            )
        return scores
    except Exception:
        logger.exception(
            "voting layer scoring failed for %d candidate(s); "
            "falling back to all-zero scores",
            len(candidates),
        )
        return [_ZERO_SCORE] * len(candidates)


def _match_to_candidates(
    candidates: Sequence[str], evaluated_outputs: Sequence[EvaluatedOutput]
) -> list[CandidateScore]:
    """Pair each candidate with its voting result, matched by text.

    Duplicate texts each take their own result.

    Raises:
        LookupError: A candidate has no matching result.
    """
    queued: dict[str, deque[EvaluatedOutput]] = defaultdict(deque)
    for evaluated_output in evaluated_outputs:
        queued[evaluated_output.output].append(evaluated_output)

    scores: list[CandidateScore] = []
    for candidate in candidates:
        bucket = queued.get(candidate)
        if not bucket:
            raise LookupError(
                f"voting layer returned no result for candidate: {candidate!r}"
            )
        scores.append(_to_candidate_score(bucket.popleft()))
    return scores


def _to_candidate_score(evaluated_output: EvaluatedOutput) -> CandidateScore:
    """Convert one voting result to app-scale scores, flagging failures."""
    by_rubric = {
        average.rubric: average.value for average in evaluated_output.rubric_averages
    }
    raw_values = [
        by_rubric.get("relevance", -1.0),
        by_rubric.get("correctness", -1.0),
        by_rubric.get("understandability", -1.0),
        by_rubric.get("coverage", -1.0),
    ]
    # Check before `_scale` clamps -1 to 0.0.
    failed = evaluated_output.overall_score < 0 or any(
        value < 0 for value in raw_values
    )
    relevance, correctness, understandability, coverage = (
        _scale(value) for value in raw_values
    )
    return CandidateScore(
        relevance=relevance,
        correctness=correctness,
        understandability=understandability,
        coverage=coverage,
        overall=_scale(evaluated_output.overall_score),
        failed=failed,
    )


def _scale(value: float) -> float:
    """Clamp a voting value to the app's 0-5 scale."""
    return round(max(_APP_MIN, min(_APP_MAX, value)), 2)
