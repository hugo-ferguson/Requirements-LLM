from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlmodel import Session

from app.config import settings
from app.db import get_session
from app.ingest.extract import ImageExtractionError, UnsupportedFileError
from app.services.ingest import EmptyDocumentError, IngestService
from app.vector_store.embeddings import get_embedding_provider
from app.vector_store.vector_store import VectorStore

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentRead(BaseModel):
    id: int
    title: str
    filename: str | None
    extension: str | None
    created_at: datetime
    chunk_count: int


class ChunkRead(BaseModel):
    id: int
    document_id: int
    chunk_index: int | None
    content: str


def get_ingest_service(session: Session = Depends(get_session)) -> IngestService:
    return IngestService(VectorStore(session), get_embedding_provider(), settings)


@router.post("/upload", response_model=DocumentRead, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    tags: str | None = Form(None, description="Comma-separated tag names"),
    service: IngestService = Depends(get_ingest_service),
) -> DocumentRead:
    tag_names = [name.strip() for name in (tags or "").split(",") if name.strip()]

    try:
        document, chunk_count = service.ingest_file(
            file.file.read(),
            file.filename or "upload",
            title=title,
            tags=tag_names or None,
        )
    except (UnsupportedFileError, EmptyDocumentError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ImageExtractionError as error:
        # The upload itself was fine; the upstream model was not.
        raise HTTPException(status_code=502, detail=str(error)) from error

    return DocumentRead(
        id=document.id,  # type: ignore[arg-type]
        title=document.title,
        filename=document.filename,
        extension=document.extension,
        created_at=document.created_at,
        chunk_count=chunk_count,
    )


@router.get("/search", response_model=list[ChunkRead])
def search_documents(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    service: IngestService = Depends(get_ingest_service),
) -> list[ChunkRead]:
    return [
        ChunkRead(
            id=chunk.id,  # type: ignore[arg-type]
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
        )
        for chunk in service.search(q, k)
    ]
