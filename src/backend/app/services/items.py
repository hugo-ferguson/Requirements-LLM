from app.models import Item, ItemCreate
from app.repositories.items import ItemRepository


class ItemService:
    """Business logic for items, independent of HTTP and DB concerns."""

    def __init__(self, repository: ItemRepository):
        self.repository = repository

    def create_item(self, data: ItemCreate) -> Item:
        item = Item(name=data.name)
        return self.repository.create(item)

    def list_items(self) -> list[Item]:
        return self.repository.list_all()
