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

    model_config = SettingsConfigDict(env_file=_ENV_FILES, extra="ignore")


settings = Settings()
