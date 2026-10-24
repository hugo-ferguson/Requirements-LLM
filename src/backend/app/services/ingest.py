"""Turn uploaded files into embedded, searchable documents."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ingest.chunk import chunk_text
from app.ingest.extract import IMAGE_EXTENSIONS, extract_text
from app.ingest.limits import MAX_EXTRACTED_CHARS, MAX_UPLOAD_BYTES
from app.vector_store.embeddings import EmbeddingProvider
from app.vector_store.models import Document, DocumentChunk
from app.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    """Raised when no text can be extracted from a file."""


class FileTooLargeError(ValueError):
    """Raised when a file, or the text extracted from it, is over the limits."""


def _megabytes(size: int) -> str:
    """Format a size in bytes as megabytes, e.g. "10.0 MB"."""
    return f"{size / (1024 * 1024):.1f} MB"


@dataclass
class IngestResult:
    """The text extracted from an upload, and the document it became.

    `document` is None and `chunk_count` 0 for images, which aren't stored.
    """

    document: Document | None
    chunk_count: int
    text: str


class IngestService:
    """Turns an uploaded file into an embedded, searchable Document."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ):
        """Wire up storage, embeddings and the chunking settings."""
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.settings = settings

    def ingest_file(
        self,
        data: bytes,
        filename: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> IngestResult:
        """Extract, chunk and embed a file, then store it as one Document.

        Images only return their text. They're short enough to send in the
        prompt, so they aren't stored.

        Args:
            data: The file's contents.
            filename: The file's name. Its extension picks the extractor.
            title: The document's title. Defaults to `filename`.
            tags: Tag names to attach.

        Raises:
            FileTooLargeError: If the file or its text is over the limits.
            EmptyDocumentError: If the file yields no text.
            UnsupportedFileError: If the extension isn't supported.
            ImageExtractionError: If the vision model fails or returns no text.
        """
        if len(data) > MAX_UPLOAD_BYTES:
            # Real size unknown: the route reads only one byte past the limit.
            raise FileTooLargeError(
                f"{filename} is larger than the "
                f"{_megabytes(MAX_UPLOAD_BYTES)} upload limit."
            )

        # Log each stage, so a slow upload doesn't look stuck.
        started = time.perf_counter()
        logger.info("ingest %r: started (%d KB)", filename, len(data) // 1024)
        text = extract_text(data, filename)
        extension = Path(filename).suffix.lower()
        logger.info(
            "ingest %r: extracted %d characters in %.1fs",
            filename,
            len(text),
            time.perf_counter() - started,
        )

        if len(text) > MAX_EXTRACTED_CHARS:
            raise FileTooLargeError(
                f"{filename} has {len(text):,} characters of text; the limit is "
                f"{MAX_EXTRACTED_CHARS:,}. Split it, or upload only the relevant pages."
            )

        if extension in IMAGE_EXTENSIONS:
            if not text.strip():
                raise EmptyDocumentError(
                    f"No text could be extracted from {filename!r}"
                )
            return IngestResult(document=None, chunk_count=0, text=text)

        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)

        if not chunks:
            raise EmptyDocumentError(f"No text could be extracted from {filename!r}")

        # Embed first, so a failure stores nothing.
        logger.info("ingest %r: embedding %d chunk(s)", filename, len(chunks))
        embeddings = self.embedding_provider.embed_texts(chunks)

        document = self.vector_store.create_document(
            source_type="upload",
            title=title or filename,
            filename=filename,
            extension=extension or None,
            tags=tags,
        )

        self.vector_store.add_chunks(
            [
                DocumentChunk(
                    document_id=document.id,  # type: ignore[arg-type]
                    content=content,
                    chunk_index=index,
                    embedding=embedding,
                )
                for index, (content, embedding) in enumerate(
                    zip(chunks, embeddings, strict=False)
                )
            ]
        )

        logger.info(
            "ingest %r: stored as document %s, %d chunk(s), %.1fs total",
            filename,
            document.id,
            len(chunks),
            time.perf_counter() - started,
        )
        return IngestResult(document=document, chunk_count=len(chunks), text=text)

    def delete_document(self, document_id: int) -> bool:
        """Remove an ingested document, returning False if it doesn't exist."""
        return self.vector_store.delete_document(document_id)

    def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Return the `top_k` chunks most similar to `query`."""
        return self.vector_store.similarity_search(
            self.embedding_provider.embed_query(query), top_k
        )
