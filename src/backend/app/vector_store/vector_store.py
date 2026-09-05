from __future__ import annotations

from sqlalchemy import delete
from sqlmodel import Session, select

from app.vector_store.models import Document, DocumentChunk, Tag


class VectorStore:
	"""Accesses the database for vector search and document operations."""

	def __init__(self, session: Session):
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
		"""
		Creates a Document row.
		"""
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
		"""
		Returns an existing Tag with the given name, or creates a new one if it
		doesn't exist.
		"""
		existing_tag = self.session.exec(
			select(Tag).where(Tag.name == name)
		).first()

		return existing_tag if existing_tag else Tag(name=name)

	def add_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
		"""
		Adds a single document chunk to the database.
		"""
		self.session.add(chunk)
		self.session.commit()
		self.session.refresh(chunk)

		return chunk

	def add_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
		"""
		Adds multiple document chunks to the database in a single transaction.
		"""
		self.session.add_all(chunks)
		self.session.commit()
		for chunk in chunks:
			self.session.refresh(chunk)

		return chunks

	def delete_document(self, document_id: int) -> bool:
		"""
		Removes a document, its chunks and its tag links.

		Returns False if no such document exists. The Tag rows themselves are
		left alone — they're shared between documents, so deleting one
		document shouldn't take a tag away from the others.
		"""
		document = self.session.get(Document, document_id)

		if document is None:
			return False

		# The chunks carry the embeddings, so delete them in one statement
		# rather than loading every vector into memory just to discard it.
		self.session.execute(
			delete(DocumentChunk).where(
				DocumentChunk.document_id == document_id  # type: ignore[arg-type]
			)
		)
		# Clearing the relationship removes the document_tag rows for us.
		document.tags = []
		self.session.delete(document)
		self.session.commit()

		return True

	def similarity_search(
			self, query_embedding: list[float], k: int = 5
		) -> list[DocumentChunk]:
		"""
		Finds the k most similar document chunks to the given query embedding.
		"""
		statement = (
			select(DocumentChunk)
			.order_by(DocumentChunk.embedding.cosine_distance(query_embedding)) # type: ignore
			.limit(k)
		)

		return list(self.session.exec(statement).all())
