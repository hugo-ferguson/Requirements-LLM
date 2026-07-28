from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from app.config import Settings


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


class OllamaEmbeddingProvider(EmbeddingProvider):
	"""
	Calls a local or self-hosted Ollama server's batch embeddings endpoint.
	"""

	TIMEOUT_SECONDS = 30.0

	def __init__(
			self, 
			model: str, 
			base_url: str = "http://localhost:11434", 
			timeout: float = TIMEOUT_SECONDS
		):
		self._model = model
		self._client = httpx.Client(base_url=base_url, timeout=timeout)
		self._dimension = len(self.embed_texts(["_"])[0])

	@property
	def dimension(self) -> int:
		return self._dimension

	def embed_texts(self, texts: list[str]) -> list[list[float]]:
		response = self._client.post(
			"/api/embed", json={"model": self._model, "input": texts}
		)

		response.raise_for_status()

		return response.json()["embeddings"]


class APIEmbeddingProvider(EmbeddingProvider):
	"""Calls an OpenAI-compatible /v1/embeddings endpoint."""

	TIMEOUT_SECONDS = 30.0

	def __init__(
			self, 
			model: str, 
			api_key: str, 
			base_url: str | None = None, 
			timeout: float = TIMEOUT_SECONDS
		):
		from openai import OpenAI

		self._model = model
		self._client = OpenAI(
			api_key=api_key, 
			base_url=base_url, 
			timeout=timeout
		)
		self._dimension = len(self.embed_texts(["_"])[0])

	@property
	def dimension(self) -> int:
		return self._dimension

	def embed_texts(self, texts: list[str]) -> list[list[float]]:
		response = self._client.embeddings.create(
			model=self._model, input=texts
		)
		
		return [item.embedding for item in response.data]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
	"""
	Constructs the EmbeddingProvider selected by environment variable.
	"""

	provider = _construct_provider(settings)

	# Check that the dimension of the embedding provider (model) matches the 
	# settings.
	# Basically, this is an unsolved problem at the moment because pgvector
	# needs a fixed vector dimension, but this can change if the model changes.
	# For now, just drop document chunk and make sure your dimension setting
	# matches the model - then the table will be created to match.
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

	if settings.embedding_provider == "ollama":
		return OllamaEmbeddingProvider(
			model=settings.embedding_model,
			base_url=settings.ollama_base_url
		)

	if settings.embedding_provider == "api":
		if not settings.embedding_api_key:
			raise ValueError(
				"EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=api"
			)

		return APIEmbeddingProvider(
			model=settings.embedding_model,
			api_key=settings.embedding_api_key,
			base_url=settings.embedding_api_base_url,
		)

	raise ValueError(f"Unknown provider: {settings.embedding_provider!r}")
