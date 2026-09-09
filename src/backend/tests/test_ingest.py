import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingest.chunk import chunk_text
from app.ingest.extract import ImageExtractionError, UnsupportedFileError, extract_text
from app.main import app
from app.routes.documents import get_ingest_service
from app.services.ingest import EmptyDocumentError, IngestResult, IngestService
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
    def __init__(self):
        self.embed_calls = 0

    def embed_texts(self, texts):
        self.embed_calls += 1
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0, 1.0]


@pytest.fixture(name="service")
def service_fixture() -> tuple[IngestService, FakeVectorStore]:
    store = FakeVectorStore()
    settings = Settings(chunk_size=200, chunk_overlap=40)
    return IngestService(store, FakeEmbeddingProvider(), settings), store


def test_an_image_is_transcribed_but_never_stored(service, monkeypatch):
    """
    A screenshot's transcript is small enough to go straight into a prompt, so
    it must not reach the vector store or the embedding provider.
    """
    ingest_service, store = service
    monkeypatch.setattr(
        "app.services.ingest.extract_text",
        lambda data, filename, settings: "Mass Actions. Select all in section.",
    )

    result = ingest_service.ingest_file(b"\x89PNG-bytes", "001-1.png")

    assert result.text == "Mass Actions. Select all in section."
    assert result.document is None
    assert result.chunk_count == 0
    assert store.created_with == {}
    assert store.chunks == []
    assert ingest_service.embedding_provider.embed_calls == 0


def test_a_pdf_is_still_chunked_and_embedded(service, monkeypatch):
    ingest_service, store = service
    monkeypatch.setattr(
        "app.services.ingest.extract_text",
        lambda data, filename, settings: "Requirement text. " * 60,
    )

    result = ingest_service.ingest_file(b"%PDF-bytes", "spec.pdf")

    assert result.document is not None
    assert result.chunk_count == len(store.chunks) > 0
    assert ingest_service.embedding_provider.embed_calls == 1


def test_an_image_the_model_returns_nothing_for_is_rejected(service, monkeypatch):
    ingest_service, _ = service
    monkeypatch.setattr(
        "app.services.ingest.extract_text",
        lambda data, filename, settings: "   ",
    )

    with pytest.raises(EmptyDocumentError):
        ingest_service.ingest_file(b"bytes", "blank.png")


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

    result = ingest_service.ingest_file(
        ("Requirement text. " * 60).encode(),
        "spec.txt",
        tags=["reqs"],
    )

    assert result.document.title == "spec.txt"
    assert result.text.startswith("Requirement text.")
    assert store.created_with["extension"] == ".txt"
    assert store.created_with["tags"] == ["reqs"]
    assert result.chunk_count == len(store.chunks)
    assert [chunk.chunk_index for chunk in store.chunks] == list(
        range(result.chunk_count)
    )
    assert all(chunk.embedding for chunk in store.chunks)


def test_file_without_text_is_rejected(service):
    ingest_service, _ = service

    with pytest.raises(EmptyDocumentError):
        ingest_service.ingest_file(b"   ", "empty.txt")


class StubIngestService:
    """Returns a canned result so the route test needs no DB or model."""

    def __init__(self, error: Exception | None = None, stores_document: bool = True):
        self.error = error
        self.stores_document = stores_document
        self.called_with: tuple[bytes, str] | None = None
        self.deleted: int | None = None

    def delete_document(self, document_id):
        self.deleted = document_id
        return document_id == 7

    def ingest_file(self, data, filename, *, title=None, tags=None):
        self.called_with = (data, filename)
        if self.error is not None:
            raise self.error
        if self.stores_document:
            return IngestResult(
                document=Document(
                    id=7,
                    source_type="upload",
                    title=title or filename,
                    filename=filename,
                    extension=".pdf",
                ),
                chunk_count=2,
                text="Login screen with a username field.",
            )
        return IngestResult(
            document=None,
            chunk_count=0,
            text="Login screen with a username field.",
        )


def _upload(stub, filename="screenshot.png"):
    """
    Posts a file with the ingest service stubbed out.

    Uses its own TestClient rather than the shared `client` fixture: the app's
    lifespan creates the pgvector extension, and nothing here touches the
    database.
    """
    app.dependency_overrides[get_ingest_service] = lambda: stub
    try:
        return TestClient(app).post(
            "/documents/upload",
            files={"file": (filename, b"fake-bytes", "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_ingest_service, None)


def test_upload_returns_the_extracted_text_and_the_stored_document_id():
    stub = StubIngestService()

    response = _upload(stub, filename="spec.pdf")

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == 7
    assert body["filename"] == "spec.pdf"
    assert body["chunk_count"] == 2
    assert body["text"] == "Login screen with a username field."
    assert stub.called_with == (b"fake-bytes", "spec.pdf")


def test_upload_of_an_unstored_file_reports_a_null_document_id():
    stub = StubIngestService(stores_document=False)

    response = _upload(stub)

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] is None
    assert body["chunk_count"] == 0
    assert body["filename"] == "screenshot.png"
    assert body["extension"] == ".png"
    assert body["text"] == "Login screen with a username field."


def test_upload_reports_an_unreadable_image_as_a_bad_gateway():
    stub = StubIngestService(ImageExtractionError("vision model returned no text"))

    response = _upload(stub)

    assert response.status_code == 502
    assert "no text" in response.json()["detail"]


def test_upload_rejects_an_unsupported_file():
    stub = StubIngestService(UnsupportedFileError("Cannot extract text from 'a.zip'"))

    response = _upload(stub, filename="a.zip")

    assert response.status_code == 422
    assert "Cannot extract text" in response.json()["detail"]


def _delete(stub, document_id):
    app.dependency_overrides[get_ingest_service] = lambda: stub
    try:
        return TestClient(app).delete(f"/documents/{document_id}")
    finally:
        app.dependency_overrides.pop(get_ingest_service, None)


def test_delete_removes_an_abandoned_document():
    stub = StubIngestService()

    response = _delete(stub, 7)

    assert response.status_code == 204
    assert stub.deleted == 7


def test_delete_404s_for_a_document_that_is_already_gone():
    stub = StubIngestService()

    response = _delete(stub, 999)

    assert response.status_code == 404
