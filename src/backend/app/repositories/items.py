from sqlmodel import Session, select

from app.models import Item


class ItemRepository:
    """Owns all direct database access for Item rows."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, item: Item) -> Item:
        self.session.add(item)
        self.session.commit()
        self.session.refresh(item)
        return item

    def list_all(self) -> list[Item]:
        return list(self.session.exec(select(Item)).all())
