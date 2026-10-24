"""HTTP routes for a session's UAT test cases."""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models_uat_cases import (
    UatApplyApprovedRequest,
    UatCase,
    UatCaseGroup,
    UatCaseGroupsResult,
    UatCaseStatusUpdate,
    UatCaseTextUpdate,
    UatRegenerateSelectedRequest,
    UatRegenerateSelectedResponse,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.agents import GenerationError
from app.services.uat_cases import UatCaseService

router = APIRouter(prefix="/sessions/{session_id}/uat-cases", tags=["uat-cases"])


def get_uat_case_service(session: Session = Depends(get_session)) -> UatCaseService:
    """Build the UAT case service for one request."""
    return UatCaseService(
        SessionRepository(session),
        UatCaseRepository(session),
        AcceptanceCriteriaRepository(session),
    )


@router.get("", response_model=UatCaseGroupsResult)
def list_uat_cases(
    session_id: int, service: UatCaseService = Depends(get_uat_case_service)
) -> UatCaseGroupsResult:
    """Return the session's test cases, grouped by AC."""
    result = service.list_for_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.post("/generate", response_model=UatCaseGroupsResult)
async def generate_uat_cases(
    session_id: int, service: UatCaseService = Depends(get_uat_case_service)
) -> UatCaseGroupsResult:
    """Generate test cases for each accepted AC, replacing the old ones."""
    try:
        result = await service.generate(session_id)
    except GenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
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
    """Save a reviewer's edit to a test case's text."""
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
    """Set a test case's review status."""
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
    """Propose one refined version of a test case, scored but not saved."""
    try:
        result = service.regenerate_selected(session_id, uat_id, data)
    except GenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
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
    """Replace a test case with the approved candidates.

    Returns the parent AC's updated group.
    """
    try:
        group = service.apply_approved(session_id, uat_id, data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if group is None:
        raise HTTPException(status_code=404, detail="UAT case not found")
    return group


@router.post("/{uat_id}/alternatives/{candidate_id}/select", response_model=UatCase)
def select_uat_alternative(
    session_id: int,
    uat_id: int,
    candidate_id: int,
    service: UatCaseService = Depends(get_uat_case_service),
) -> UatCase:
    """Show another model's version of a test case in its place."""
    result = service.select_alternative(session_id, uat_id, candidate_id)
    if result is None:
        raise HTTPException(status_code=404, detail="UAT case or alternative not found")
    return result
