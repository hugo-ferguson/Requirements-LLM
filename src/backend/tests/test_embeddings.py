import pytest
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.vector_store.embeddings import EmbeddingProvider, build_embedding_provider
from app.vector_store.models import Document, DocumentChunk, Tag
from app.vector_store.vector_store import VectorStore


@pytest.fixture(scope="module")
def provider() -> EmbeddingProvider:
	return build_embedding_provider(settings)


def test_create_document_and_embed_chunk(provider: EmbeddingProvider) -> None:
	with Session(engine) as session:
		store = VectorStore(session)

		document = store.create_document(
			source_type="user_story", title="Login"
		)
		assert document.id is not None
		chunk = None

		try:
			content = (
				"As a user, I want to log in with my email and password so I "
				"can access my dashboard."
			)

			[embedding] = provider.embed_texts([content])

			chunk = store.add_chunk(DocumentChunk(
				document_id=document.id, 
				content=content, 
				embedding=embedding))

			assert chunk.id is not None
			assert chunk.document_id == document.id
			assert len(chunk.embedding) == provider.dimension

			found = store.similarity_search(
				provider.embed_query("logging in"), 
				k=1
			)

			assert found
			assert found[0].id == chunk.id

		finally:
			if chunk is not None:
				session.delete(chunk)

			session.delete(document)
			session.commit()


def test_delete_document_removes_its_chunks_but_keeps_shared_tags(
	provider: EmbeddingProvider,
) -> None:
	with Session(engine) as session:
		store = VectorStore(session)

		document = store.create_document(
			source_type="upload", title="Abandoned upload", tags=["scratch"]
		)
		assert document.id is not None
		document_id = document.id

		try:
			[embedding] = provider.embed_texts(["A discarded screenshot."])
			store.add_chunk(DocumentChunk(
				document_id=document_id,
				content="A discarded screenshot.",
				embedding=embedding,
			))

			assert store.delete_document(document_id) is True

			assert session.get(Document, document_id) is None
			remaining = session.exec(
				select(DocumentChunk).where(
					DocumentChunk.document_id == document_id
				)
			).all()
			assert remaining == []

			# Tags outlive the documents that referenced them.
			assert session.exec(
				select(Tag).where(Tag.name == "scratch")
			).first() is not None

			# Deleting again is a miss, not an error.
			assert store.delete_document(document_id) is False

		finally:
			tag = session.exec(select(Tag).where(Tag.name == "scratch")).first()
			if tag is not None:
				session.delete(tag)
				session.commit()
