from pathlib import Path

from app.config import Settings
from app.ingest.chunk import chunk_text
from app.ingest.extract import extract_text
from app.vector_store.embeddings import EmbeddingProvider
from app.vector_store.models import Document, DocumentChunk
from app.vector_store.vector_store import VectorStore


class EmptyDocumentError(ValueError):
    """Raised when a file yields no text to embed."""


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
    ) -> tuple[Document, int]:
        """
        Extracts, chunks and embeds a file, then stores it as one Document.

        Returns the stored document and its chunk count.
        """
        text = extract_text(data, filename, self.settings)
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

        return document, len(chunks)

    def search(self, query: str, k: int = 5) -> list[DocumentChunk]:
        """Finds the k chunks most similar to a natural-language query."""
        return self.vector_store.similarity_search(
            self.embedding_provider.embed_query(query), k
        )
