"""Table and API models for the example items resource."""

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class ItemBase(SQLModel):
    """Fields shared by every item model."""

    name: str


class Item(ItemBase, table=True):
    """A stored item row."""

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))


class ItemCreate(ItemBase):
    """Request body for creating an item."""


class ItemRead(ItemBase):
    """An item as returned by the API."""

    id: int
    created_at: datetime
