"""Text extraction, chunking, ingest and the document upload route."""

from io import BytesIO
from typing import cast

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingest.chunk import chunk_text
from app.ingest.extract import (
    MAX_IMAGE_DIMENSION,
    ImageExtractionError,
    UnsupportedFileError,
    _prepare_image,
    extract_text,
)
from app.main import app
from app.routes.documents import get_ingest_service
from app.services.ingest import (
    EmptyDocumentError,
    FileTooLargeError,
    IngestResult,
    IngestService,
)
from app.vector_store.embeddings import EmbeddingProvider
from app.vector_store.models import Document, DocumentChunk
from app.vector_store.vector_store import VectorStore


class FakeVectorStore:
    """Stands in for VectorStore so the tests need no database."""

    def __init__(self):
        """Start with no document and no chunks."""
        self.created_with: dict = {}
        self.chunks: list[DocumentChunk] = []

    def create_document(self, **kwargs):
        """Record the arguments and return document 1."""
        self.created_with = kwargs
        return Document(
            id=1,
            source_type=kwargs["source_type"],
            title=kwargs["title"],
            filename=kwargs["filename"],
            extension=kwargs["extension"],
        )

    def add_chunks(self, chunks):
        """Keep the chunks for the test to inspect."""
        self.chunks = chunks
        return chunks

    def similarity_search(self, query_embedding, k=5):
        """Return the first `k` stored chunks."""
        return self.chunks[:k]


class FakeEmbeddingProvider:
    """Returns tiny length-based vectors and counts embed calls."""

    def __init__(self):
        """Start with no embed calls counted."""
        self.embed_calls = 0

    def embed_texts(self, texts):
        """Count the call and return one small vector per text."""
        self.embed_calls += 1
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def embed_query(self, text):
        """Return a small vector for `text`."""
        return [float(len(text)), 0.0, 1.0]


@pytest.fixture(name="service")
def service_fixture() -> tuple[IngestService, FakeVectorStore]:
    """Return an ingest service on fakes, plus its fake store."""
    store = FakeVectorStore()
    settings = Settings(chunk_size=200, chunk_overlap=40)
    service = IngestService(
        cast(VectorStore, store),
        cast(EmbeddingProvider, FakeEmbeddingProvider()),
        settings,
    )
    return service, store


def test_an_image_is_transcribed_but_never_stored(service, monkeypatch):
    """
    A screenshot's transcript is small enough to go straight into a
    prompt, so it must not reach the vector store or the embedding
    provider.
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


def test_chunking_terminates_when_a_break_falls_exactly_overlap_past_the_start():
    # A line break exactly `overlap` characters into the next chunk must
    # not make that chunk restart where it began and append forever. A
    # real PDF once grew the backend until Docker's VM ran out of
    # memory.
    text = ("x" * 849) + "\n" + ("y" * 149) + "\n" + ("z" * 3000)

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    assert 1 < len(chunks) < 10
    assert chunks[-1].endswith("z")


def test_blank_text_produces_no_chunks():
    assert chunk_text("   \n  ") == []


def test_unknown_extension_is_rejected():
    with pytest.raises(UnsupportedFileError):
        extract_text(b"data", "archive.zip", Settings())


def _gif(width: int, height: int) -> bytes:
    """Return a blank GIF of the given size."""
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, "GIF")
    return buffer.getvalue()


def test_a_small_image_is_sent_untouched_with_its_own_media_type():
    data = _gif(100, 100)

    assert _prepare_image(data, ".gif") == (data, "image/gif")


def test_a_resized_image_is_labelled_with_the_format_it_was_saved_in():
    from PIL import Image

    resized, mime = _prepare_image(_gif(MAX_IMAGE_DIMENSION * 2, 10), ".gif")

    image = Image.open(BytesIO(resized))
    assert (image.format, mime) == ("PNG", "image/png")
    assert max(image.size) == MAX_IMAGE_DIMENSION


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


def test_a_file_over_the_byte_limit_is_rejected_before_extraction(service, monkeypatch):
    ingest_service, store = service
    monkeypatch.setattr("app.services.ingest.MAX_UPLOAD_BYTES", 100)

    def fail_extract(data, filename, settings):
        raise AssertionError("an oversized file must not be extracted")

    monkeypatch.setattr("app.services.ingest.extract_text", fail_extract)

    with pytest.raises(FileTooLargeError, match="larger than"):
        ingest_service.ingest_file(b"x" * 101, "big.pdf")
    assert store.created_with == {}


def test_a_file_with_too_much_text_is_rejected_and_not_stored(service, monkeypatch):
    """The text limit is what bounds LLM cost: attachments are resent every turn."""
    ingest_service, store = service
    monkeypatch.setattr("app.services.ingest.MAX_EXTRACTED_CHARS", 50)

    with pytest.raises(FileTooLargeError, match="51 characters"):
        ingest_service.ingest_file(b"y" * 51, "long.txt")
    assert store.created_with == {}
    assert ingest_service.embedding_provider.embed_calls == 0


def test_a_file_exactly_at_both_limits_is_accepted(service, monkeypatch):
    ingest_service, _ = service
    monkeypatch.setattr("app.services.ingest.MAX_UPLOAD_BYTES", 50)
    monkeypatch.setattr("app.services.ingest.MAX_EXTRACTED_CHARS", 50)

    result = ingest_service.ingest_file(b"z" * 50, "fits.txt")

    assert result.document is not None


class StubIngestService:
    """Returns a canned result so route tests need no DB or model."""

    def __init__(self, error: Exception | None = None, stores_document: bool = True):
        """Raise `error` on ingest, or return a canned result."""
        self.error = error
        self.stores_document = stores_document
        self.called_with: tuple[bytes, str] | None = None
        self.deleted: int | None = None

    def delete_document(self, document_id):
        """Record the id; only document 7 exists."""
        self.deleted = document_id
        return document_id == 7

    def ingest_file(self, data, filename, *, title=None, tags=None):
        """Record the call and return the canned result."""
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
    """Post a file with the ingest service stubbed out.

    Uses a bare TestClient rather than the shared `client` fixture,
    which runs the app's lifespan and so needs the database; nothing
    here touches it.
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
    stub = StubIngestService(ImageExtractionError(
        "vision model returned no text"))

    response = _upload(stub)

    assert response.status_code == 502
    assert "no text" in response.json()["detail"]


def test_upload_reports_an_oversized_file_as_413():
    stub = StubIngestService(FileTooLargeError(
        "big.pdf is larger than the 10.0 MB upload limit."))

    response = _upload(stub, filename="big.pdf")

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


def test_upload_reads_no_more_than_one_byte_past_the_limit(monkeypatch):
    monkeypatch.setattr("app.routes.documents.MAX_UPLOAD_BYTES", 4)
    stub = StubIngestService()

    _upload(stub, filename="spec.pdf")

    # b"fake-bytes" is 10 bytes; the route hands over limit + 1 and the
    # service's own check rejects it.
    assert stub.called_with == (b"fake-", "spec.pdf")


def test_upload_rejects_an_unsupported_file():
    stub = StubIngestService(UnsupportedFileError(
        "Cannot extract text from 'a.zip'"))

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
