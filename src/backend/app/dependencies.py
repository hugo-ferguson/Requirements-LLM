"""FastAPI dependencies shared between route modules.

Kept here so route modules don't import each other.
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
    """Build the document ingest service for one request."""
    return IngestService(VectorStore(session), get_embedding_provider(), settings)


def get_generation_service(
    ingest_service: IngestService = Depends(get_ingest_service),
) -> GenerationService:
    """Build the generation service for one request.

    The ingest service gives the agents RAG context from uploads (backlog R11).
    """
    return GenerationService(settings, ingest_service)
