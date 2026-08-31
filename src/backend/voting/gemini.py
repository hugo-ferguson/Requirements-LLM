from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from voting.combined_provider import evaluate_with_combined_model
from voting.models import CombinedVote, EvaluationInput, PrometheusVote, VotingResult


load_dotenv()

COMBINED_PROMPT = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer" / "combined.txt"


def _build_generation_config() -> dict[str, Any]:
    return {
        "temperature": 0.1,
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "correctness": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 1, "maximum": 5},
                        "feedback": {"type": "STRING"},
                    },
                    "required": ["score", "feedback"],
                },
                "coverage": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 1, "maximum": 5},
                        "feedback": {"type": "STRING"},
                    },
                    "required": ["score", "feedback"],
                },
                "relevance": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 1, "maximum": 5},
                        "feedback": {"type": "STRING"},
                    },
                    "required": ["score", "feedback"],
                },
                "understandability": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 1, "maximum": 5},
                        "feedback": {"type": "STRING"},
                    },
                    "required": ["score", "feedback"],
                },
            },
            "required": ["correctness", "coverage", "relevance", "understandability"],
        },
    }


def _normalize_gemini_schema(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            normalized[normalized_key] = _normalize_gemini_schema(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_gemini_schema(item) for item in value]
    return value


def _coerce_score(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return -1
        for pattern in (
            r"(?i)\b(?:score|rating)\s*[:=]\s*(-?\d+)\s*(?:/\s*5|out\s+of\s+5)?\b",
            r"(?i)\b(-?\d+)\s*(?:/\s*5|out\s+of\s+5)\b",
            r"(?i)\b(-?\d+)\b",
        ):
            match = re.search(pattern, text)
            if match:
                try:
                    result = int(float(match.group(1)))
                except ValueError:
                    continue
                if -1 <= result <= 5:
                    return result
    return -1


def _parse_prometheus_vote(raw_value: Any) -> PrometheusVote:
    if isinstance(raw_value, PrometheusVote):
        return raw_value

    if isinstance(raw_value, dict):
        score = raw_value.get("score")
        feedback = raw_value.get("feedback")
        if score is not None:
            return PrometheusVote(score=_coerce_score(score), feedback=str(feedback or ""))

    text = str(raw_value).strip()
    if not text:
        return PrometheusVote(score=-1, feedback="No rubric feedback returned.")

    score_match = re.search(r"(?i)\b(?:score|rating)\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*(?:/\s*5|out\s+of\s+5)?\b", text)
    if not score_match:
        score_match = re.search(r"(?i)\b(-?\d+(?:\.\d+)?)\s*(?:/\s*5|out\s+of\s+5)\b", text)
    if not score_match:
        score_match = re.search(r"(?i)\b(-?\d+(?:\.\d+)?)\b", text)

    feed_match = re.search(r"(?is)(?:feedback|reason|explanation)\s*[:=]\s*(.*)$", text)
    score = _coerce_score(score_match.group(1)) if score_match else -1
    feedback = feed_match.group(1).strip() if feed_match else text
    return PrometheusVote(score=score, feedback=feedback)


def _fallback_vote_for_text(raw_text: str) -> CombinedVote:
    rubric_names = ("correctness", "coverage", "relevance", "understandability")
    fallback: dict[str, PrometheusVote] = {}
    for rubric in rubric_names:
        section = raw_text
        label_match = re.search(rf"(?is)\b{rubric}\b\s*[:\-]\s*(.*?)(?=\n\s*(?:correctness|coverage|relevance|understandability)\s*[:\-]|\Z)", raw_text)
        if label_match:
            section = label_match.group(1).strip()
            score = _coerce_score(section)
            feedback = section if score == -1 else section
            fallback[rubric] = PrometheusVote(score=score, feedback=feedback)
        else:
            fallback[rubric] = PrometheusVote(score=-1, feedback=raw_text.strip() or "Gemini returned no structured rubric payload.")
    return CombinedVote(**fallback)


def _coerce_rubric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_gemini_schema(payload)

    if all(name in normalized for name in ("correctness", "coverage", "relevance", "understandability")):
        return {
            rubric: _parse_prometheus_vote(normalized[rubric])
            for rubric in ("correctness", "coverage", "relevance", "understandability")
        }

    feedback = normalized.get("feedback")
    if isinstance(feedback, dict):
        rubric_payload = {
            rubric: _parse_prometheus_vote(feedback[rubric])
            for rubric in ("correctness", "coverage", "relevance", "understandability")
            if rubric in feedback
        }
        if len(rubric_payload) == 4:
            return rubric_payload

    nested = normalized.get("result")
    if isinstance(nested, dict):
        return _coerce_rubric_payload(nested)

    coerced = {
        rubric: _parse_prometheus_vote(normalized.get(rubric))
        for rubric in ("correctness", "coverage", "relevance", "understandability")
        if rubric in normalized and normalized.get(rubric) not in ({}, None)
    }
    if len(coerced) == 4:
        return coerced
    return {}


def _extract_rubric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _coerce_rubric_payload(payload)


def _parse_response_text(raw_text: str) -> CombinedVote:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        payload = json.loads(cleaned)
    except (TypeError, ValueError):
        return _fallback_vote_for_text(cleaned)

    if not isinstance(payload, dict):
        return _fallback_vote_for_text(cleaned)

    coerced = _extract_rubric_payload(payload)
    if len(coerced) != 4:
        return _fallback_vote_for_text(cleaned)
    return CombinedVote.model_validate(coerced)


class GeminiCombinedClient:
    def __init__(
        self,
        provider_name: str,
        default_model: str,
        model_env: str,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        max_concurrency: int = 1,
        base_backoff_seconds: float = 1.0,
    ) -> None:
        self.provider_name = provider_name
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your local .env file before running Gemini evaluation.")
        self.base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.model = model or os.getenv(model_env, default_model)
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.base_backoff_seconds = base_backoff_seconds
        self._rate_limit_backoff = base_backoff_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _post_with_backoff(self, *, request: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        last_error: httpx.HTTPStatusError | None = None
        for attempt in range(self.max_retries + 1):
            async with self._semaphore:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        result = await client.post(
                            f"{self.base_url}/models/{self.model}:generateContent",
                            json=request,
                            headers=headers,
                        )
                    if result.status_code == 429 or 500 <= result.status_code < 600:
                        result.raise_for_status()
                    return result
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response is None:
                        raise
                    status = exc.response.status_code
                    if status == 429 or 500 <= status < 600:
                        if attempt >= self.max_retries:
                            raise
                        delay = min(self._rate_limit_backoff * (2 ** attempt), 30.0)
                        delay += random.uniform(0.25, 1.25)
                        self._rate_limit_backoff = min(self._rate_limit_backoff * 2, 30.0)
                        await asyncio.sleep(delay)
                        continue
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini evaluation failed with no response")

    async def evaluate(self, *, instruction: str, response: str) -> CombinedVote:
        prompt = COMBINED_PROMPT.read_text(encoding="utf-8")
        prompt = prompt.replace("{orig_instruction}", instruction)
        prompt = prompt.replace("{orig_response}", response)
        request = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": _build_generation_config(),
        }
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }

        result = await self._post_with_backoff(request=request, headers=headers)
        body: dict[str, Any] = result.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        raw_text = "".join(text_parts).strip()
        if not raw_text:
            raise ValueError("Gemini returned no message content")

        self._rate_limit_backoff = self.base_backoff_seconds
        return _parse_response_text(raw_text)


async def evaluate_with_gemini(evaluation_input: EvaluationInput) -> VotingResult:
    client = GeminiCombinedClient(
        provider_name="Gemini",
        default_model="gemini-flash-latest",
        model_env="GEMINI_MODEL",
    )
    return await evaluate_with_combined_model(evaluation_input, client)
