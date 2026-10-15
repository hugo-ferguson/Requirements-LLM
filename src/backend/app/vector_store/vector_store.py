"""Database access for documents, chunks and similarity search."""

from __future__ import annotations

from sqlalchemy import delete
from sqlmodel import Session, select

from app.vector_store.models import Document, DocumentChunk, Tag


class VectorStore:
    """Database access for documents, chunks and vector search."""

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
        """Create a document row, creating any tags it names."""
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
        existing_tag = self.session.exec(
            select(Tag).where(Tag.name == name)
        ).first()

        return existing_tag if existing_tag else Tag(name=name)

    def add_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Store one document chunk."""
        self.session.add(chunk)
        self.session.commit()
        self.session.refresh(chunk)

        return chunk

    def add_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Store several document chunks in a single transaction."""
        self.session.add_all(chunks)
        self.session.commit()
        for chunk in chunks:
            self.session.refresh(chunk)

        return chunks

    def delete_document(self, document_id: int) -> bool:
        """Remove a document, its chunks and its tag links.

        Return False if no such document exists. The Tag rows themselves
        are left alone — they're shared between documents, so deleting
        one document shouldn't take a tag away from the others.
        """
        document = self.session.get(Document, document_id)

        if document is None:
            return False

        # The chunks carry the embeddings, so delete them in one
        # statement rather than loading every vector into memory just to
        # discard it.
        self.session.execute(
            delete(DocumentChunk).where(
                # type: ignore[arg-type]
                DocumentChunk.document_id == document_id
            )
        )
        # Clearing the relationship removes the document_tag rows for
        # us.
        document.tags = []
        self.session.delete(document)
        self.session.commit()

        return True

    def similarity_search(
        self, query_embedding: list[float], k: int = 5
    ) -> list[DocumentChunk]:
        """Find the `k` chunks closest to `query_embedding`."""
        statement = (
            select(DocumentChunk)
            # type: ignore
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(k)
        )

        return list(self.session.exec(statement).all())
