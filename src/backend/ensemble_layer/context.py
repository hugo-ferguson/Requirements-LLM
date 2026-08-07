from __future__ import annotations

import os
from dataclasses import dataclass, field

from .models import EnsembleConfig, Provider


@dataclass
class VectorStoreStub:
    """
    Placeholder for the Vector Store shown in the architecture diagram.
    Will be replaced with a real embedding/retrieval implementation
    (e.g. ChromaDB, pgvector) once that layer is wired up.
    """
    _store: dict = field(default_factory=dict)

    def similarity_search(self, query: str, k: int = 3) -> list[str]:
        return []

    def add_document(self, doc_id: str, text: str) -> None:
        self._store[doc_id] = text


_PROVIDER_ENV_VAR = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GEMINI: "GEMINI_API_KEY",
}


@dataclass
class RunContext:
    """
    Injected into every Generation Agent at call time.

    Attributes:
        config:       Ensemble run configuration (agent list, etc.)
        vector_store: Shared vector store for RAG context retrieval.
        run_id:       Correlates all agents in the same ensemble run.
    """
    config: EnsembleConfig
    vector_store: VectorStoreStub
    run_id: str | None = None

    @classmethod
    def build(
        cls,
        config: EnsembleConfig,
        run_id: str | None = None,
    ) -> "RunContext":
        """
        Factory — validates that the required API keys are set for every
        provider referenced in the agent configs. PydanticAI reads these
        env vars directly when it instantiates its model clients.
        """
        providers_needed = {a.provider for a in config.agents}
        missing = [
            _PROVIDER_ENV_VAR[p] for p in providers_needed
            if not os.environ.get(_PROVIDER_ENV_VAR[p])
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required API key env var(s): {', '.join(missing)}"
            )

        return cls(
            config=config,
            vector_store=VectorStoreStub(),
            run_id=run_id,
        )