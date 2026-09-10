from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import litellm

from app.config import Settings, settings


class EmbeddingProvider(ABC):
	"""
	Turns text into fixed-length vectors for similarity search.

	Extended by concrete embedding providers that may use a local model, API,
	etc.
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

	def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
		from fastembed import TextEmbedding

		self._model = TextEmbedding(model_name=model_name)
		self._dimension = len(self.embed_texts(["_"])[0])

	@property
	def dimension(self) -> int:
		return self._dimension

	def embed_texts(self, texts: list[str]) -> list[list[float]]:
		return [vector.tolist() for vector in self._model.embed(texts)]


class LiteLLMEmbeddingProvider(EmbeddingProvider):
	"""Calls any LiteLLM-supported embedding endpoint."""

	def __init__(self, model: str, api_key: str | None = None):
		self._model = model
		self._api_key = api_key
		self._dimension = len(self.embed_texts(["_"])[0])

	@property
	def dimension(self) -> int:
		return self._dimension

	def embed_texts(self, texts: list[str]) -> list[list[float]]:
		kwargs: dict = {"model": self._model, "input": texts}
		if self._api_key:
			kwargs["api_key"] = self._api_key
		response = litellm.embedding(**kwargs)
		return [item["embedding"] for item in response.data]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
	"""
	Returns the process-wide provider, building it on first use.

	Construction loads a model and issues a test embedding, so it must not
	happen per request.
	"""
	return build_embedding_provider(settings)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
	"""
	Constructs the EmbeddingProvider selected by environment variable.
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
	if settings.embedding_provider == "local":
		return LocalEmbeddingProvider(model_name=settings.embedding_model)

	if settings.embedding_provider == "litellm":
		return LiteLLMEmbeddingProvider(
			model=settings.embedding_model,
			api_key=settings.embedding_api_key,
		)

	raise ValueError(f"Unknown provider: {settings.embedding_provider!r}")
