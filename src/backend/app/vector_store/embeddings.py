"""Embedding providers that turn text into vectors for search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, settings


class EmbeddingProvider(ABC):
    """Turns text into fixed-length vectors for similarity search."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this provider returns."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_texts([text])[0]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Runs a small embedding model locally with fastembed."""

    # Smaller than fastembed's default of 256, which ran out of memory on large PDFs.
    BATCH_SIZE = 16

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """Load `model_name` and measure the length of its vectors."""
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dimension = len(self.embed_texts(["_"])[0])

    @property
    def dimension(self) -> int:
        """Length of the vectors this model returns."""
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of BATCH_SIZE, preserving order."""
        return [
            vector.tolist()
            for vector in self._model.embed(texts, batch_size=self.BATCH_SIZE)
        ]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the shared provider, building it on first use."""
    return build_embedding_provider(settings)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured provider and check its vector length.

    Raises:
        ValueError: If the provider is unknown or doesn't match `EMBEDDING_DIM`.
    """
    provider = _construct_provider(settings)

    if provider.dimension != settings.embedding_dim:
        raise ValueError(
            f"Embedding provider {settings.embedding_provider!r} with model "
            f"{settings.embedding_model!r} produces {provider.dimension}-dim "
            f"vectors. EMBEDDING_DIM is set to {settings.embedding_dim}. "
            "Update EMBEDDING_DIM to match, or choose a different EMBEDDING_MODEL."
        )

    return provider


def _construct_provider(settings: Settings) -> EmbeddingProvider:
    """Build the provider that `EMBEDDING_PROVIDER` names."""
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(model_name=settings.embedding_model)

    raise ValueError(f"Unknown provider: {settings.embedding_provider!r}")
