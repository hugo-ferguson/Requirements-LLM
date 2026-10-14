"""Table models for stored documents, their tags and chunks."""

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from app.config import settings


class DocumentTag(SQLModel, table=True):
    """Link table for the many-to-many Document and Tag relationship."""

    __tablename__: str = "document_tag"  # type: ignore

    document_id: int = Field(foreign_key="document.id", primary_key=True)
    tag_id: int = Field(foreign_key="tag.id", primary_key=True)


class Tag(SQLModel, table=True):
    """A label that categorises documents for users and LLMs."""

    __tablename__: str = "tag"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

    documents: list["Document"] = Relationship(
        back_populates="tags",
        link_model=DocumentTag
    )


class Document(SQLModel, table=True):
    """A document that has been ingested into the system."""

    __tablename__: str = "document"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    source_type: str = Field(index=True)
    title: str
    filename: str | None = None
    extension: str | None = None
    source_uri: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    tags: list[Tag] = Relationship(
        back_populates="documents",
        link_model=DocumentTag
    )
    chunks: list["DocumentChunk"] = Relationship(back_populates="document")


class DocumentChunk(SQLModel, table=True):
    """One embedded chunk of an ingested document."""

    __tablename__: str = "document_chunk"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    content: str
    chunk_index: int | None = None
    embedding: list[float] = Field(
        sa_column=Column(Vector(settings.embedding_dim))
    )
    chunk_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    document: Document | None = Relationship(back_populates="chunks")
