from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at src/backend/app/, so walk up to the repo root.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _BACKEND_DIR.parent
_REPO_ROOT = _SRC_DIR.parent

# Read every .env we might reasonably find, most general first — pydantic
# lets later files win, so a backend-local .env still overrides the shared
# one. Resolving them absolutely means settings no longer depend on the
# directory uvicorn happened to be launched from.
_ENV_FILES = (
    _REPO_ROOT / ".env",
    _SRC_DIR / ".env",
    _BACKEND_DIR / ".env",
)


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/app"
    sql_echo: bool = False
    cors_origins: list[str] = ["http://localhost:5173"]

    embedding_provider: Literal["local", "litellm"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_api_key: str | None = None

    vision_model: str = "ollama/qwen2.5vl:7b"

    # The model that writes acceptance criteria. Any model string PydanticAI
    # knows works here, so changing provider is a config change, not a code
    # change. Note the "google-gla:" prefix used in the PydanticAI spike was
    # renamed to "google:" in pydantic-ai 2.x.
    llm_model: str = "google:gemini-3-flash-preview"
    gemini_api_key: str | None = None

    chunk_size: int = 1000
    chunk_overlap: int = 150


    # --- Ensemble generation layer ---
    # Roster of generation agents (backlog R8). Relative paths resolve
    # against the backend working directory.
    generation_roster_path: str = "config/generation_agents.json"
    # Per-agent wall-clock limit. Keeps one slow model from eating the whole
    # 60s budget in R6.
    generation_timeout_seconds: float = 45.0
    # Upper bound on criteria requested from each agent.
    generation_max_criteria: int = 8
    # Number of RAG chunks injected into each agent's prompt (R11).
    rag_top_k: int = 5
    # Set false to skip the voting layer and return unscored candidates -
    # useful when iterating on prompts without paying for the voters.
    generation_enable_voting: bool = True
    # Two-pass generation: one agent fixes the titles, the rest write their own
    # given/when/then against them, so candidates for the same behaviour can be
    # compared directly instead of pooled into an ungrouped union. Costs one
    # extra sequential pass; set false to restore the single-pass ensemble.
    generation_title_anchored: bool = True
    # Voting judges: comma-separated names from VOTING_PROVIDERS. Every judge
    # scores every candidate and their rubric scores are averaged, so one
    # model's taste (or its preference for its own family's wording) does not
    # pick every winner alone. A judge that fails is averaged out.
    voting_judges: str = "claude"

    @property
    def voting_judge_names(self) -> list[str]:
        names = [name.strip().lower() for name in self.voting_judges.split(",")]
        return [name for name in names if name] or ["claude"]

    model_config = SettingsConfigDict(env_file=_ENV_FILES, extra="ignore")


settings = Settings()
