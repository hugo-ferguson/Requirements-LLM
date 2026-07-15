from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db import get_session
from app.models import ItemCreate, ItemRead
from app.repositories.items import ItemRepository
from app.services.items import ItemService

router = APIRouter(prefix="/items", tags=["items"])


def get_item_service(session: Session = Depends(get_session)) -> ItemService:
    return ItemService(ItemRepository(session))


@router.get("", response_model=list[ItemRead])
def list_items(service: ItemService = Depends(get_item_service)) -> list[ItemRead]:
    return service.list_items()


@router.post("", response_model=ItemRead, status_code=201)
def create_item(
    data: ItemCreate, service: ItemService = Depends(get_item_service)
) -> ItemRead:
    return service.create_item(data)
