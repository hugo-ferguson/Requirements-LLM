"""Shared FastAPI dependency providers.

These live here rather than inside a route module so that routes never have to
import each other just to reuse a dependency. `documents`, `sessions` and
`acceptance_criteria` all need the ingest and generation services.
"""

from fastapi import Depends
from sqlmodel import Session

from app.config import settings
from app.db import get_session
from app.services.generation import GenerationService
from app.services.ingest import IngestService
from app.vector_store.embeddings import get_embedding_provider
from app.vector_store.vector_store import VectorStore


def get_ingest_service(session: Session = Depends(get_session)) -> IngestService:
    return IngestService(VectorStore(session), get_embedding_provider(), settings)


def get_generation_service(
    ingest_service: IngestService = Depends(get_ingest_service),
) -> GenerationService:
    """Build the ensemble generation service for one request.

    The ingest service is passed in as the RAG retriever, so generation agents
    receive project-specific context chunks (backlog R11).
    """
    return GenerationService(settings, ingest_service)
