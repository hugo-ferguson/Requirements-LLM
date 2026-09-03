from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from voting.combined_provider import evaluate_with_combined_model
from voting.models import CombinedVote, EvaluationInput, VotingResult


load_dotenv()

COMBINED_PROMPT = Path(__file__).resolve().parent.parent / "ai_prompts" / "voting_layer" / "combined.txt"


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
        self.client = Anthropic(api_key=self.api_key, timeout=self.timeout)

    async def evaluate(self, *, instruction: str, response: str) -> CombinedVote:
        prompt = COMBINED_PROMPT.read_text(encoding="utf-8")
        prompt = prompt.replace("{orig_instruction}", instruction)
        prompt = prompt.replace("{orig_response}", response)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,
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

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()

        import json as _json
        import re as _re

        match = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
        if match:
            cleaned = match.group(0)

        payload = _json.loads(cleaned)
        return CombinedVote.model_validate(payload)


async def evaluate_with_claude(evaluation_input: EvaluationInput) -> VotingResult:
    client = ClaudeCombinedClient(
        provider_name="Claude",
        default_model="claude-sonnet-4-6",
        model_env="CLAUDE_MODEL",
    )
    return await evaluate_with_combined_model(evaluation_input, client)
