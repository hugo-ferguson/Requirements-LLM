"""Business logic for the example items resource."""

from app.models import Item, ItemCreate
from app.repositories.items import ItemRepository


class ItemService:
    """Business logic for items, independent of HTTP and DB concerns."""

    def __init__(self, repository: ItemRepository):
        """Use `repository` for all item storage."""
        self.repository = repository

    def create_item(self, data: ItemCreate) -> Item:
        """Create an item from a request body."""
        item = Item(name=data.name)
        return self.repository.create(item)

    def list_items(self) -> list[Item]:
        """Return every item."""
        return self.repository.list_all()
