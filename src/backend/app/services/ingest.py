from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ingest.chunk import chunk_text
from app.ingest.extract import IMAGE_EXTENSIONS, extract_text
from app.vector_store.embeddings import EmbeddingProvider
from app.vector_store.models import Document, DocumentChunk
from app.vector_store.vector_store import VectorStore


class EmptyDocumentError(ValueError):
    """Raised when a file yields no text to embed."""


@dataclass
class IngestResult:
    """
    The text extracted from an upload, and the Document it was stored as.

    `document` is None for files that are read but deliberately not stored —
    see IngestService.ingest_file.
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
        """
        Extracts, chunks and embeds a file, then stores it as one Document.

        Images are the exception: their text goes straight back to the caller
        and is never stored. A screenshot transcribes to a few hundred tokens
        — measured across a real screenshot set, five images came to 789
        tokens, one chunk each — so it fits in a prompt many times over and
        embedding it only adds a retrieval hop and rows nothing queries.
        Documents that can actually outgrow a prompt (PDFs, long text) are
        still chunked and embedded.

        Returns the extracted text, plus the stored document and its chunk
        count when the file was stored.
        """
        text = extract_text(data, filename, self.settings)
        extension = Path(filename).suffix.lower()

        if extension in IMAGE_EXTENSIONS:
            if not text.strip():
                raise EmptyDocumentError(
                    f"No text could be extracted from {filename!r}"
                )
            return IngestResult(document=None, chunk_count=0, text=text)

        chunks = chunk_text(
            text, self.settings.chunk_size, self.settings.chunk_overlap
        )

        if not chunks:
            raise EmptyDocumentError(
                f"No text could be extracted from {filename!r}"
            )

        # Embed before writing anything, so a provider failure leaves no
        # half-ingested document behind.
        embeddings = self.embedding_provider.embed_texts(chunks)

        document = self.vector_store.create_document(
            source_type="upload",
            title=title or filename,
            filename=filename,
            extension=Path(filename).suffix.lower() or None,
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
                    zip(chunks, embeddings)
                )
            ]
        )

        return IngestResult(document=document, chunk_count=len(chunks), text=text)

    def delete_document(self, document_id: int) -> bool:
        """
        Removes an ingested document. Returns False if it doesn't exist.

        Used when an attachment is discarded before it's sent, so abandoned
        uploads don't linger in the search index.
        """
        return self.vector_store.delete_document(document_id)

    def search(self, query: str, k: int = 5) -> list[DocumentChunk]:
        """Finds the k chunks most similar to a natural-language query."""
        return self.vector_store.similarity_search(
            self.embedding_provider.embed_query(query), k
        )
