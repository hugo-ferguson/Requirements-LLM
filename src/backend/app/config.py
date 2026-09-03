from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/app"
    sql_echo: bool = False
    cors_origins: list[str] = ["http://localhost:5173"]

    embedding_provider: Literal["local", "litellm"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_api_key: str | None = None

    vision_model: str = "ollama/qwen2.5vl:7b"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
