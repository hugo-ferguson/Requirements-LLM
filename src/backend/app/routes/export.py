"""HTTP route for exporting a session's approved requirements."""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models_export import ExportResponse
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.export import ExportService

router = APIRouter(prefix="/sessions/{session_id}/export", tags=["export"])


def get_export_service(session: Session = Depends(get_session)) -> ExportService:
    """Build the export service for one request."""
    return ExportService(
        SessionRepository(session),
        AcceptanceCriteriaRepository(session),
        UatCaseRepository(session),
    )


@router.get("", response_model=ExportResponse)
def get_export(
    session_id: int, service: ExportService = Depends(get_export_service)
) -> ExportResponse:
    """Return the session's accepted ACs and UAT cases as a JSON document.

    `warnings` lists what's left out, so the user can confirm first.
    """
    result = service.build(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result
