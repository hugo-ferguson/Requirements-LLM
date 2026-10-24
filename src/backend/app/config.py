"""Application settings, read from the environment and .env files."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Walk up from src/backend/app/ to the repo root.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _BACKEND_DIR.parent
_REPO_ROOT = _SRC_DIR.parent

# Later files win, so a backend .env overrides the shared ones.
_ENV_FILES = (
    _REPO_ROOT / ".env",
    _SRC_DIR / ".env",
    _BACKEND_DIR / ".env",
)


class Settings(BaseSettings):
    """Every runtime setting, overridable by environment variable."""

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/app"
    sql_echo: bool = False
    cors_origins: list[str] = ["http://localhost:5173"]

    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # config/models.json sets which model does chat, vision, generation and judging.
    models_config_path: str = str(_BACKEND_DIR / "config" / "models.json")

    gemini_api_key: str | None = None

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Timeout for each agent.
    generation_timeout_seconds: float = 45.0
    # Maximum criteria per agent.
    generation_max_criteria: int = 8
    # RAG chunks added to each agent's prompt (R11).
    rag_top_k: int = 5
    # Off: skip the judges and return unscored candidates.
    generation_enable_voting: bool = True
    # One agent writes the titles, then the others write an AC for each title.
    # Off: every agent writes its own.
    generation_title_anchored: bool = True

    model_config = SettingsConfigDict(env_file=_ENV_FILES, extra="ignore")


settings = Settings()
