from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models_acceptance_criteria import (
    AcceptanceCriterionStatusUpdate,
    AcceptanceCriterionTextUpdate,
    ApplyApprovedRequest,
    RegenerateSelectedRequest,
    RegenerateSelectedResponse,
)
from app.models_conversation import AcceptanceCriterion, GenerateResult
from app.models_session import MessageRead
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.acceptance_criteria import AcceptanceCriteriaService

router = APIRouter(
    prefix="/sessions/{session_id}/acceptance-criteria", tags=["acceptance-criteria"]
)


def get_acceptance_criteria_service(
    session: Session = Depends(get_session),
) -> AcceptanceCriteriaService:
    return AcceptanceCriteriaService(
        SessionRepository(session),
        AcceptanceCriteriaRepository(session),
        MessageRepository(session),
        UatCaseRepository(session),
    )


@router.get("", response_model=GenerateResult)
def list_acceptance_criteria(
    session_id: int, service: AcceptanceCriteriaService = Depends(get_acceptance_criteria_service)
) -> GenerateResult:
    items = service.list_items(session_id)
    if items is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return GenerateResult(acceptance_criteria=items)


@router.patch("/{ac_id}", response_model=AcceptanceCriterion)
def update_acceptance_criterion_text(
    session_id: int,
    ac_id: int,
    data: AcceptanceCriterionTextUpdate,
    service: AcceptanceCriteriaService = Depends(get_acceptance_criteria_service),
) -> AcceptanceCriterion:
    updated = service.update_text(session_id, ac_id, data)
    if updated is None:
        raise HTTPException(status_code=404, detail="Acceptance criterion not found")
    return updated


@router.patch("/{ac_id}/status", response_model=AcceptanceCriterion)
def update_acceptance_criterion_status(
    session_id: int,
    ac_id: int,
    data: AcceptanceCriterionStatusUpdate,
    service: AcceptanceCriteriaService = Depends(get_acceptance_criteria_service),
) -> AcceptanceCriterion:
    updated = service.update_status(session_id, ac_id, data)
    if updated is None:
        raise HTTPException(status_code=404, detail="Acceptance criterion not found")
    return updated


@router.post("/{ac_id}/regenerate", response_model=RegenerateSelectedResponse)
def regenerate_selected(
    session_id: int,
    ac_id: int,
    data: RegenerateSelectedRequest,
    service: AcceptanceCriteriaService = Depends(get_acceptance_criteria_service),
) -> RegenerateSelectedResponse:
    result = service.regenerate_selected(session_id, ac_id, data)
    if result is None:
        raise HTTPException(status_code=404, detail="Acceptance criterion not found")
    return result


@router.post("/{ac_id}/apply-approved", response_model=GenerateResult)
def apply_approved(
    session_id: int,
    ac_id: int,
    data: ApplyApprovedRequest,
    service: AcceptanceCriteriaService = Depends(get_acceptance_criteria_service),
) -> GenerateResult:
    try:
        items = service.apply_approved(session_id, ac_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if items is None:
        raise HTTPException(status_code=404, detail="Acceptance criterion not found")
    return GenerateResult(acceptance_criteria=items)


@router.post("/regenerate-all-kickoff", response_model=MessageRead, status_code=201)
def regenerate_all_kickoff(
    session_id: int, service: AcceptanceCriteriaService = Depends(get_acceptance_criteria_service)
) -> MessageRead:
    message = service.regenerate_all_kickoff(session_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return message
