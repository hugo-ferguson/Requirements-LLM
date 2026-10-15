"""Embedding providers that turn text into vectors for search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, settings


class EmbeddingProvider(ABC):
    """Turns text into fixed-length vectors for similarity search.

    Extended by concrete embedding providers that may use a local model,
    API, etc.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this provider returns."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        return self.embed_texts([text])[0]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Runs a small embedding model locally with fastembed."""

    # fastembed's default batch is 256 chunks. The model's activations
    # grow with batch size x sequence length squared, so a large PDF
    # briefly needed gigabytes — enough to exhaust Docker's WSL VM and
    # take the whole stack down mid-upload. Small batches give identical
    # vectors at a fraction of the peak memory, for a negligible cost in
    # speed on CPU.
    BATCH_SIZE = 16

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """Load `model_name` and measure the size of its vectors."""
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dimension = len(self.embed_texts(["_"])[0])

    @property
    def dimension(self) -> int:
        """Length of the vectors this model returns."""
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in small batches, preserving order."""
        return [
            vector.tolist()
            for vector in self._model.embed(texts, batch_size=self.BATCH_SIZE)
        ]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide provider, building it on first use.

    Construction loads a model and issues a test embedding, so it must
    not happen per request.
    """
    return build_embedding_provider(settings)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured provider and check its vector size.

    Raise `ValueError` when the model's vectors don't match
    `EMBEDDING_DIM`, or the provider name is unknown.
    """
    provider = _construct_provider(settings)

    if provider.dimension != settings.embedding_dim:
        raise ValueError(
            f"Embedding provider {settings.embedding_provider!r} with model "
            f"{settings.embedding_model!r} produces {provider.dimension}-dim vectors."
            f"EMBEDDING_DIM is set to {settings.embedding_dim}. "
            f"Update EMBEDDING_DIM to match, or choose a different EMBEDDING_MODEL."
        )

    return provider


def _construct_provider(settings: Settings) -> EmbeddingProvider:
    # Map the EMBEDDING_PROVIDER setting to a provider class.
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(model_name=settings.embedding_model)

    raise ValueError(f"Unknown provider: {settings.embedding_provider!r}")
