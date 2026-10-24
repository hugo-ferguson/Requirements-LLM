"""Table models for stored documents, their tags and chunks."""

from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from app.config import settings


class DocumentTag(SQLModel, table=True):
    """Link table between documents and tags."""

    __tablename__: str = "document_tag"  # type: ignore

    document_id: int = Field(foreign_key="document.id", primary_key=True)
    tag_id: int = Field(foreign_key="tag.id", primary_key=True)


class Tag(SQLModel, table=True):
    """A label that categorises documents."""

    __tablename__: str = "tag"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

    documents: list["Document"] = Relationship(
        back_populates="tags", link_model=DocumentTag
    )


class Document(SQLModel, table=True):
    """An ingested document."""

    __tablename__: str = "document"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    source_type: str = Field(
        index=True, description='Where the document came from, e.g. "upload".'
    )
    title: str
    filename: str | None = None
    extension: str | None = Field(
        default=None, description='Lower-case extension with its dot, e.g. ".pdf".'
    )
    source_uri: str | None = Field(
        default=None, description="Location of the original, if any."
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    tags: list[Tag] = Relationship(back_populates="documents", link_model=DocumentTag)
    chunks: list["DocumentChunk"] = Relationship(back_populates="document")


class DocumentChunk(SQLModel, table=True):
    """One embedded chunk of an ingested document."""

    __tablename__: str = "document_chunk"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    content: str
    chunk_index: int | None = Field(
        default=None, description="Position in the document, from 0."
    )
    embedding: list[float] = Field(sa_column=Column(Vector(settings.embedding_dim)))
    chunk_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Free-form details about the chunk.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    document: Document | None = Relationship(back_populates="chunks")
