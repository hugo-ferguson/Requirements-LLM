"""Database access for documents, chunks and similarity search."""

from __future__ import annotations

from sqlalchemy import delete
from sqlmodel import Session, select

from app.vector_store.models import Document, DocumentChunk, Tag


class VectorStore:
    """Stores documents and their embedded chunks, and searches the chunks."""

    def __init__(self, session: Session):
        """Bind the store to one database session."""
        self.session = session

    def create_document(
        self,
        *,
        source_type: str,
        title: str,
        filename: str | None = None,
        extension: str | None = None,
        source_uri: str | None = None,
        tags: list[str] | None = None,
    ) -> Document:
        """Save a new document, creating any missing tags."""
        document = Document(
            source_type=source_type,
            title=title,
            filename=filename,
            extension=extension,
            source_uri=source_uri,
            tags=[self._get_or_create_tag(name) for name in (tags or [])],
        )

        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)

        return document

    def _get_or_create_tag(self, name: str) -> Tag:
        """Return the tag called `name`, or a new unsaved one."""
        existing_tag = self.session.exec(select(Tag).where(Tag.name == name)).first()

        return existing_tag or Tag(name=name)

    def add_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Store one document chunk."""
        self.session.add(chunk)
        self.session.commit()
        self.session.refresh(chunk)

        return chunk

    def add_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Store several document chunks in one transaction."""
        self.session.add_all(chunks)
        self.session.commit()
        for chunk in chunks:
            self.session.refresh(chunk)

        return chunks

    def delete_document(self, document_id: int) -> bool:
        """Remove a document, its chunks and its tag links, keeping the tags.

        Returns:
            False if the document didn't exist.
        """
        document = self.session.get(Document, document_id)

        if document is None:
            return False

        # Bulk delete, so the embeddings aren't loaded first.
        self.session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id  # type: ignore[arg-type]
            )
        )
        # Clearing the relationship deletes the document_tag rows.
        document.tags = []
        self.session.delete(document)
        self.session.commit()

        return True

    def similarity_search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[DocumentChunk]:
        """Return the `top_k` closest chunks, closest first."""
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)  # type: ignore[attr-defined]
        statement = select(DocumentChunk).order_by(distance).limit(top_k)

        return list(self.session.exec(statement).all())
