"""HTTP routes for uploading, searching and deleting documents."""

from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from pydantic import BaseModel

from app.dependencies import get_ingest_service
from app.ingest.extract import ImageExtractionError, UnsupportedFileError
from app.ingest.limits import MAX_UPLOAD_BYTES
from app.services.ingest import EmptyDocumentError, FileTooLargeError, IngestService

router = APIRouter(prefix="/documents", tags=["documents"])


class UploadRead(BaseModel):
    """An upload's outcome: always text, sometimes a stored document."""

    # None for images, which aren't stored.
    document_id: int | None
    title: str
    filename: str | None
    extension: str | None
    chunk_count: int
    # Lets the caller attach the text without a second request.
    text: str


class ChunkRead(BaseModel):
    """One stored chunk of a document, as returned by search."""

    id: int
    document_id: int
    chunk_index: int | None
    content: str


@router.post("/upload", response_model=UploadRead, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    tags: str | None = Form(None, description="Comma-separated tag names"),
    ingest_service: IngestService = Depends(get_ingest_service),
) -> UploadRead:
    """Extract a file's text and, unless it's an image, index it for search.

    Images aren't stored, so `document_id` is null. Responds 413 if too large,
    422 if unsupported or empty, and 502 if the vision model fails.
    """
    filename = file.filename or "upload"
    tag_names = [name.strip() for name in (tags or "").split(",") if name.strip()]

    try:
        result = ingest_service.ingest_file(
            # One byte past the limit is enough to reject an oversized file.
            file.file.read(MAX_UPLOAD_BYTES + 1),
            filename,
            title=title,
            tags=tag_names or None,
        )
    except FileTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except (UnsupportedFileError, EmptyDocumentError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ImageExtractionError as error:
        # The vision model failed, not the upload.
        raise HTTPException(status_code=502, detail=str(error)) from error

    document = result.document
    if document is None:
        return UploadRead(
            document_id=None,
            title=title or filename,
            filename=filename,
            extension=Path(filename).suffix.lower() or None,
            chunk_count=result.chunk_count,
            text=result.text,
        )
    return UploadRead(
        document_id=document.id,
        title=document.title,
        filename=document.filename,
        extension=document.extension,
        chunk_count=result.chunk_count,
        text=result.text,
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    ingest_service: IngestService = Depends(get_ingest_service),
) -> Response:
    """Delete a stored document and its chunks, or respond 404 if it's gone."""
    if not ingest_service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


@router.get("/search", response_model=list[ChunkRead])
def search_documents(
    query: str = Query(..., alias="q", min_length=1),
    top_k: int = Query(5, alias="k", ge=1, le=50),
    ingest_service: IngestService = Depends(get_ingest_service),
) -> list[ChunkRead]:
    """Return the `k` stored chunks most similar to the query `q`."""
    return [
        ChunkRead(
            id=chunk.id,  # type: ignore[arg-type]
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
        )
        for chunk in ingest_service.search(query, top_k)
    ]
