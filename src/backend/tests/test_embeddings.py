import pytest
from sqlmodel import Session

from app.config import settings
from app.db import engine
from app.vector_store.embeddings import EmbeddingProvider, build_embedding_provider
from app.vector_store.models import DocumentChunk
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
