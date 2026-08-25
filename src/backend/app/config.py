from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/app"
    sql_echo: bool = False
    cors_origins: list[str] = ["http://localhost:5173"]

    embedding_provider: Literal["local", "ollama", "api"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    ollama_base_url: str = "http://localhost:11434"
    embedding_api_key: str | None = None
    embedding_api_base_url: str | None = None

    # Local Ollama vision model that reads text out of images during ingest.
    ollama_vision_model: str = "qwen2.5vl:7b"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
