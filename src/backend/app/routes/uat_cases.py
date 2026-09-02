from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models_uat_cases import (
    UatCase,
    UatCaseGroup,
    UatCaseGroupsResult,
    UatCaseStatusUpdate,
    UatCaseTextUpdate,
    UatApplyApprovedRequest,
    UatRegenerateSelectedRequest,
    UatRegenerateSelectedResponse,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.uat_cases import UatCaseService

router = APIRouter(prefix="/sessions/{session_id}/uat-cases", tags=["uat-cases"])


def get_uat_case_service(session: Session = Depends(get_session)) -> UatCaseService:
    return UatCaseService(
        SessionRepository(session),
        UatCaseRepository(session),
        AcceptanceCriteriaRepository(session),
    )


@router.get("", response_model=UatCaseGroupsResult)
def list_uat_cases(
    session_id: int, service: UatCaseService = Depends(get_uat_case_service)
) -> UatCaseGroupsResult:
    result = service.list_items(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.post("/generate", response_model=UatCaseGroupsResult)
def generate_uat_cases(
    session_id: int, service: UatCaseService = Depends(get_uat_case_service)
) -> UatCaseGroupsResult:
    result = service.generate(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.patch("/{uat_id}", response_model=UatCase)
def update_uat_case_text(
    session_id: int,
    uat_id: int,
    data: UatCaseTextUpdate,
    service: UatCaseService = Depends(get_uat_case_service),
) -> UatCase:
    updated = service.update_text(session_id, uat_id, data)
    if updated is None:
        raise HTTPException(status_code=404, detail="UAT case not found")
    return updated


@router.patch("/{uat_id}/status", response_model=UatCase)
def update_uat_case_status(
    session_id: int,
    uat_id: int,
    data: UatCaseStatusUpdate,
    service: UatCaseService = Depends(get_uat_case_service),
) -> UatCase:
    updated = service.update_status(session_id, uat_id, data)
    if updated is None:
        raise HTTPException(status_code=404, detail="UAT case not found")
    return updated


@router.post("/{uat_id}/regenerate", response_model=UatRegenerateSelectedResponse)
def regenerate_selected(
    session_id: int,
    uat_id: int,
    data: UatRegenerateSelectedRequest,
    service: UatCaseService = Depends(get_uat_case_service),
) -> UatRegenerateSelectedResponse:
    result = service.regenerate_selected(session_id, uat_id, data)
    if result is None:
        raise HTTPException(status_code=404, detail="UAT case not found")
    return result


@router.post("/{uat_id}/apply-approved", response_model=UatCaseGroup)
def apply_approved(
    session_id: int,
    uat_id: int,
    data: UatApplyApprovedRequest,
    service: UatCaseService = Depends(get_uat_case_service),
) -> UatCaseGroup:
    try:
        group = service.apply_approved(session_id, uat_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if group is None:
        raise HTTPException(status_code=404, detail="UAT case not found")
    return group
