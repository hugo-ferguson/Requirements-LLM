import pytest

from app.config import Settings
from app.ingest.chunk import chunk_text
from app.ingest.extract import UnsupportedFileError, extract_text
from app.services.ingest import EmptyDocumentError, IngestService
from app.vector_store.models import Document, DocumentChunk


class FakeVectorStore:
    """Stands in for VectorStore so the tests need no database."""

    def __init__(self):
        self.created_with: dict = {}
        self.chunks: list[DocumentChunk] = []

    def create_document(self, **kwargs):
        self.created_with = kwargs
        return Document(
            id=1,
            source_type=kwargs["source_type"],
            title=kwargs["title"],
            filename=kwargs["filename"],
            extension=kwargs["extension"],
        )

    def add_chunks(self, chunks):
        self.chunks = chunks
        return chunks

    def similarity_search(self, query_embedding, k=5):
        return self.chunks[:k]


class FakeEmbeddingProvider:
    def embed_texts(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0, 1.0]


@pytest.fixture(name="service")
def service_fixture() -> tuple[IngestService, FakeVectorStore]:
    store = FakeVectorStore()
    settings = Settings(chunk_size=200, chunk_overlap=40)
    return IngestService(store, FakeEmbeddingProvider(), settings), store


def test_chunks_stay_within_size_and_cover_the_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(6))

    chunks = chunk_text(text, chunk_size=500, overlap=80)

    assert chunks
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert "Paragraph 5." in chunks[-1]


def test_chunking_terminates_without_whitespace_boundaries():
    assert chunk_text("x" * 2500, chunk_size=500, overlap=80)


def test_blank_text_produces_no_chunks():
    assert chunk_text("   \n  ") == []


def test_unknown_extension_is_rejected():
    with pytest.raises(UnsupportedFileError):
        extract_text(b"data", "archive.zip", Settings())


def test_ingest_stores_one_chunk_per_embedding(service):
    ingest_service, store = service

    document, chunk_count = ingest_service.ingest_file(
        ("Requirement text. " * 60).encode(),
        "spec.txt",
        tags=["reqs"],
    )

    assert document.title == "spec.txt"
    assert store.created_with["extension"] == ".txt"
    assert store.created_with["tags"] == ["reqs"]
    assert chunk_count == len(store.chunks)
    assert [chunk.chunk_index for chunk in store.chunks] == list(
        range(chunk_count)
    )
    assert all(chunk.embedding for chunk in store.chunks)


def test_file_without_text_is_rejected(service):
    ingest_service, _ = service

    with pytest.raises(EmptyDocumentError):
        ingest_service.ingest_file(b"   ", "empty.txt")
