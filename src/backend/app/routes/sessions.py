from fastapi import APIRouter, Depends, HTTPException, Response
from sqlmodel import Session

from app.db import get_session
from app.models_conversation import GenerateResult
from app.models_session import (
    MessageCreate,
    MessageRead,
    SessionCreate,
    SessionDetail,
    SessionSummary,
    SessionUpdate,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.conversation import (
    ConversationService,
    EmptyConversationError,
    GenerationError,
)
from app.services.sessions import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


def get_session_service(session: Session = Depends(get_session)) -> SessionService:
    return SessionService(
        SessionRepository(session),
        MessageRepository(session),
        ConversationService(),
        AcceptanceCriteriaRepository(session),
        UatCaseRepository(session),
    )


@router.post("", response_model=SessionSummary, status_code=201)
def create_session(
    data: SessionCreate, service: SessionService = Depends(get_session_service)
) -> SessionSummary:
    return service.create_session(data)


@router.get("", response_model=list[SessionSummary])
def list_sessions(service: SessionService = Depends(get_session_service)) -> list[SessionSummary]:
    return service.list_sessions()


@router.get("/{session_id}", response_model=SessionDetail)
def get_session_detail(
    session_id: int, service: SessionService = Depends(get_session_service)
) -> SessionDetail:
    detail = service.get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@router.patch("/{session_id}", response_model=SessionSummary)
def rename_session(
    session_id: int,
    data: SessionUpdate,
    service: SessionService = Depends(get_session_service),
) -> SessionSummary:
    chat_session = service.rename_session(session_id, data)
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_session


@router.delete("/{session_id}", status_code=204)
def delete_session(
    session_id: int, service: SessionService = Depends(get_session_service)
) -> Response:
    deleted = service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)


@router.post("/{session_id}/messages", response_model=MessageRead, status_code=201)
def post_message(
    session_id: int,
    data: MessageCreate,
    service: SessionService = Depends(get_session_service),
) -> MessageRead:
    reply = service.post_message(session_id, data)
    if reply is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return reply


@router.post("/{session_id}/generate", response_model=GenerateResult)
def generate(
    session_id: int, service: SessionService = Depends(get_session_service)
) -> GenerateResult:
    try:
        result = service.generate(session_id)
    except EmptyConversationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except GenerationError as error:
        # The request was fine; the upstream model was not.
        raise HTTPException(status_code=502, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result
