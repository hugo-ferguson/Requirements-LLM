from fastapi import APIRouter, Depends

from app.models_conversation import ConversationMessage, ConversationRequest, GenerateResult
from app.services.conversation import ConversationService

router = APIRouter(prefix="/conversation", tags=["conversation"])


def get_conversation_service() -> ConversationService:
    return ConversationService()


@router.post("/messages", response_model=ConversationMessage, status_code=200)
def send_message(
    data: ConversationRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationMessage:
    return service.send_message(data)


@router.post("/generate", response_model=GenerateResult, status_code=200)
def generate(
    data: ConversationRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> GenerateResult:
    return service.generate(data)
