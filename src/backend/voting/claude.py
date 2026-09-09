from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from voting.combined_provider import evaluate_with_combined_model
from voting.gemini import _parse_response_text
from voting.models import CombinedVote, EvaluationInput, VotingResult


load_dotenv()

COMBINED_PROMPT = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer" / "combined.txt"

_RESPONSE_SCHEMA_INSTRUCTION = (
    "\n\n###Response schema:\n"
    "Return ONLY this JSON object — no prose, no code fences:\n"
    "{\n"
    '  "correctness": {"score": <integer 1-5>, "feedback": "<text>"},\n'
    '  "coverage": {"score": <integer 1-5>, "feedback": "<text>"},\n'
    '  "relevance": {"score": <integer 1-5>, "feedback": "<text>"},\n'
    '  "understandability": {"score": <integer 1-5>, "feedback": "<text>"}\n'
    "}"
)


class ClaudeCombinedClient:
    def __init__(
        self,
        provider_name: str,
        default_model: str,
        model_env: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.provider_name = provider_name
        self.api_key = os.getenv(api_key_env) or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) is not set. Add it to your local .env file before running Claude evaluation.")
        self.model = model or os.getenv(model_env, default_model)
        self.timeout = timeout
        self.client = AsyncAnthropic(api_key=self.api_key, timeout=self.timeout)

    async def evaluate(self, *, instruction: str, response: str) -> CombinedVote:
        prompt = COMBINED_PROMPT.read_text(encoding="utf-8")
        prompt = prompt.replace("{orig_instruction}", instruction)
        prompt = prompt.replace("{orig_response}", response)
        # Gemini gets the rubric shape via a responseSchema; Claude has no
        # equivalent knob here, so spell the exact JSON out in the prompt.
        prompt += _RESPONSE_SCHEMA_INSTRUCTION

        # Note: no `temperature` — the current SDK/models drop the sampling
        # params, and the JSON-only instruction keeps output deterministic enough.
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        text_blocks: list[str] = []
        for block in getattr(message, "content", []):
            if hasattr(block, "text") and isinstance(block.text, str):
                text_blocks.append(block.text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_blocks.append(block["text"])

        raw_text = "".join(text_blocks).strip()
        if not raw_text:
            raise ValueError("Claude returned no message content")

        # Same tolerant parser Gemini uses: strips code fences, pulls the JSON
        # object out of any surrounding prose, and coerces near-miss shapes.
        return _parse_response_text(raw_text)


async def evaluate_with_claude(evaluation_input: EvaluationInput) -> VotingResult:
    client = ClaudeCombinedClient(
        provider_name="Claude",
        # Small/cheap by default; override with CLAUDE_MODEL in the env.
        default_model="claude-haiku-4-5",
        model_env="CLAUDE_MODEL",
    )
    return await evaluate_with_combined_model(evaluation_input, client)
