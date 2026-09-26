from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.config import Settings, settings as default_settings

if TYPE_CHECKING:
    from voting.models import EvaluatedOutput

logger = logging.getLogger(__name__)

# Isolated to this module: voting.voting unconditionally imports all 5
# provider modules at load time (including `from anthropic import Anthropic`
# in voting.claude), so ANY import failure here — missing optional dep,
# unrelated packaging issue — must degrade to the all-zero fallback below
# instead of taking down every route that imports this module.
try:
    from voting.models import EvaluationInput
    from voting.voting import evaluate_input as _evaluate_input
except Exception as _import_error:  # pragma: no cover - exercised via monkeypatch in tests
    EvaluationInput = None  # type: ignore[assignment,misc]
    _evaluate_input = None
    _VOTING_IMPORT_ERROR: Exception | None = _import_error
else:
    _VOTING_IMPORT_ERROR = None

# Voting rubrics score 1-5 (an int, with -1 reserved for a provider error) —
# the same 1-5 range the app's AcceptanceCriterionScores/UatCaseScores use,
# so no rescaling is needed. _APP_MIN is 0, not 1: it exists purely to catch
# the -1 error sentinel, which should read as worse than any real 1-5
# rating, not clamp up into the middle of the valid range.
_APP_MIN, _APP_MAX = 0.0, 5.0


@dataclass(frozen=True)
class CandidateScore:
    """
    One candidate's 0-10 rubric scores — structurally identical to
    AcceptanceCriterionScores/UatCaseScores so services build either
    directly from these fields.
    """

    relevance: float
    correctness: float
    understandability: float
    coverage: float
    overall: float
    # True when the voting layer never actually rated this candidate — the
    # rubric values are the -1 error sentinel clamped to 0.0, not a verdict.
    # Callers that rank candidates against each other need to tell the two
    # apart, because an unrated candidate scores 0.0 and therefore always
    # loses, which lets a scoring outage silently pick the winner.
    failed: bool = False


_ZERO_SCORE = CandidateScore(0.0, 0.0, 0.0, 0.0, 0.0, failed=True)


class Scorer(Protocol):
    def __call__(
        self,
        prompt: str,
        candidates: Sequence[str],
        *,
        reference_answer: str = "",
        settings: Settings | None = None,
    ) -> list[CandidateScore]: ...


def render_ac_candidate(title: str, given: str, when: str, then: str) -> str:
    """
    Canonical flat-text rendering of one acceptance criterion — used both as
    the voting-layer candidate `output` string and, for AC-regen/UAT
    generation prompts, as the description of the criterion under
    discussion. MUST stay the single source of truth for this shape: the
    voting layer sorts/reorders its results, so getting a score back to the
    right candidate depends on exact text equality with what was submitted.
    """
    return f"Title: {title}\nGiven: {given}\nWhen: {when}\nThen: {then}"


def render_uat_candidate(title: str, description: str) -> str:
    """Same role as `render_ac_candidate`, for a UAT test case."""
    return f"Title: {title}\nDescription: {description}"


def score_candidates(
    prompt: str,
    candidates: Sequence[str],
    *,
    reference_answer: str = "",
    settings: Settings | None = None,
) -> list[CandidateScore]:
    """
    Scores each candidate string against `prompt` via the gemini-only voting
    layer, returning one CandidateScore per candidate in the SAME ORDER
    `candidates` was given. (The voting layer itself sorts its `output` list
    by score descending, so results are matched back to candidates by exact
    text content, not position.)

    Never raises. Any failure — the voting package failing to import,
    a missing/invalid GEMINI_API_KEY, a network error, an unexpected
    response shape, or a mismatch between candidates and returned results —
    is logged and produces an all-zero CandidateScore per candidate, so a
    scoring outage never blocks AC/UAT generation or regeneration.
    """
    if not candidates:
        return []

    if _evaluate_input is None:
        logger.warning(
            "voting layer unavailable, scoring %d candidate(s) as zero: %s",
            len(candidates),
            _VOTING_IMPORT_ERROR,
        )
        return [_ZERO_SCORE] * len(candidates)

    resolved_settings = settings or default_settings
    if resolved_settings.gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
        # voting.gemini resolves its key via os.getenv + its own load_dotenv()
        # call — a resolution path independent of Settings' explicit
        # multi-file read. Bridge the two so "reuse Settings.gemini_api_key"
        # holds even if voting's directory-based .env search would miss it.
        os.environ["GEMINI_API_KEY"] = resolved_settings.gemini_api_key

    try:
        evaluation_input = EvaluationInput(
            ai="requirements-llm",
            model="claude-only",
            prompt=prompt,
            output=list(candidates),
            reference_answer=reference_answer,
            providers=["claude"],
        )
        result = asyncio.run(_evaluate_input(evaluation_input))
        scores = _match_to_candidates(candidates, result.output)

        unrated = sum(1 for score in scores if score.failed)
        if unrated and unrated != len(scores):
            # A partial failure is the dangerous case: the batch looks fine,
            # but every unrated candidate sits at 0.0 and loses to any rated
            # one regardless of how good it actually was.
            logger.warning(
                "%d of %d candidate(s) were not rated by the voting layer and "
                "scored 0.0 by default; they will lose to any rated candidate "
                "regardless of quality",
                unrated,
                len(scores),
            )

        if scores and all(score.failed for score in scores):
            # The voting layer never raises for a provider failure — it swaps in
            # the -1 sentinel, which clamps to 0.0 and looks exactly like a real
            # (terrible) score. A whole batch at zero is not a plausible
            # verdict, so flag it as configuration rather than quality.
            logger.warning(
                "every one of %d candidate(s) scored 0.0 — this is almost always a "
                "misconfigured voter rather than genuinely worthless criteria. "
                "Check that VOTING_PROVIDERS defines %r and that its model is "
                "reachable; see the provider warning logged above for the cause.",
                len(scores),
                evaluation_input.providers[0] if evaluation_input.providers else "claude",
            )
        return scores
    except Exception:
        logger.exception(
            "voting layer scoring failed for %d candidate(s); falling back to all-zero scores",
            len(candidates),
        )
        return [_ZERO_SCORE] * len(candidates)


def _match_to_candidates(
    candidates: Sequence[str], evaluated_outputs: Sequence["EvaluatedOutput"]
) -> list[CandidateScore]:
    """
    Text -> queue multimap so duplicate candidate text (unlikely, but not
    impossible if the model repeats itself) consumes one queued result per
    occurrence rather than reusing/skipping results. Raises LookupError
    (caught by the caller) if a candidate has no matching result at all.
    """
    queued: dict[str, deque] = defaultdict(deque)
    for output in evaluated_outputs:
        queued[output.output].append(output)

    scores: list[CandidateScore] = []
    for candidate in candidates:
        bucket = queued.get(candidate)
        if not bucket:
            raise LookupError(f"voting layer returned no result for candidate: {candidate!r}")
        scores.append(_to_candidate_score(bucket.popleft()))
    return scores


def _to_candidate_score(output: "EvaluatedOutput") -> CandidateScore:
    by_rubric = {avg.rubric: avg.value for avg in output.rubric_averages}
    raw = [
        by_rubric.get("relevance", -1.0),
        by_rubric.get("correctness", -1.0),
        by_rubric.get("understandability", -1.0),
        by_rubric.get("coverage", -1.0),
    ]
    # Detected BEFORE _scale, which clamps the -1 sentinel to 0.0 and makes a
    # scoring failure indistinguishable from a genuine zero.
    failed = output.overall_score < 0 or any(value < 0 for value in raw)
    relevance, correctness, understandability, coverage = (_scale(v) for v in raw)
    return CandidateScore(
        relevance=relevance,
        correctness=correctness,
        understandability=understandability,
        coverage=coverage,
        overall=_scale(output.overall_score),
        failed=failed,
    )


def _scale(value: float) -> float:
    """
    Voting's rubric values already live on the app's 0-5 scale — this just
    clamps, so voting's -1 error sentinel (or any other out-of-range value)
    lands at 0 rather than a nonsensical negative score.
    """
    return round(max(_APP_MIN, min(_APP_MAX, value)), 2)
