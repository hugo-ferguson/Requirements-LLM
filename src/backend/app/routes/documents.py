from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel
from sqlmodel import Session

from app.config import settings
from app.db import get_session
from app.ingest.extract import ImageExtractionError, UnsupportedFileError
from app.services.ingest import EmptyDocumentError, IngestService
from app.vector_store.embeddings import get_embedding_provider
from app.vector_store.vector_store import VectorStore

router = APIRouter(prefix="/documents", tags=["documents"])


class UploadRead(BaseModel):
    """The outcome of an upload: always text, sometimes a stored document."""

    # None when the file was read but not stored — images, which go straight
    # into the conversation instead of the search index.
    document_id: int | None
    title: str
    filename: str | None
    extension: str | None
    chunk_count: int
    # The extracted text, so the caller can attach it to a conversation
    # without a second round trip through /documents/search.
    text: str


class ChunkRead(BaseModel):
    id: int
    document_id: int
    chunk_index: int | None
    content: str


def get_ingest_service(session: Session = Depends(get_session)) -> IngestService:
    return IngestService(VectorStore(session), get_embedding_provider(), settings)


@router.post("/upload", response_model=UploadRead, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    tags: str | None = Form(None, description="Comma-separated tag names"),
    service: IngestService = Depends(get_ingest_service),
) -> UploadRead:
    tag_names = [name.strip() for name in (tags or "").split(",") if name.strip()]

    try:
        result = service.ingest_file(
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

    document = result.document
    filename = file.filename or "upload"
    return UploadRead(
        document_id=document.id if document else None,
        title=document.title if document else (title or filename),
        filename=document.filename if document else filename,
        extension=document.extension if document else Path(filename).suffix.lower() or None,
        chunk_count=result.chunk_count,
        text=result.text,
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    service: IngestService = Depends(get_ingest_service),
) -> Response:
    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


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
