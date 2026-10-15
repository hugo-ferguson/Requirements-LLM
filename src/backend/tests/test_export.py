"""The JSON export: approved ACs with their approved UAT cases nested under them."""

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.models_conversation import AcceptanceCriterion, AcceptanceCriterionScores
from app.models_uat_cases import UatCase, UatCaseScores
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.export import ExportService


@pytest.fixture(name="db_session")
def db_session_fixture():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _criterion(title: str) -> AcceptanceCriterion:
    scores = AcceptanceCriterionScores(relevance=4, correctness=4, understandability=4, coverage=4)
    return AcceptanceCriterion(
        id=0, title=title, given="g", when="w", then="t", scores=scores, overall_score=4,
        source_agent="agent-a",
    )


def _case(ac_id: int, title: str) -> UatCase:
    scores = UatCaseScores(relevance=3, correctness=3, understandability=3, coverage=3)
    return UatCase(
        id=-1, ac_id=ac_id, title=title, description=f"{title} steps", scores=scores,
        overall_score=3, source_agent="agent-b",
    )


def _seed(db_session: Session):
    """Login (approved, 1 of 2 cases approved), Logout (approved, no approved cases),
    Signup (pending), Reset (rejected)."""
    sessions = SessionRepository(db_session)
    ac_repo = AcceptanceCriteriaRepository(db_session)
    uat_repo = UatCaseRepository(db_session)

    chat_session = sessions.create("Auth stories")
    login, logout, signup, reset = ac_repo.persist_batch(
        chat_session.id, [_criterion(t) for t in ("Login", "Logout", "Signup", "Reset")]
    )
    ac_repo.update_status(login, "accepted")
    ac_repo.update_status(logout, "accepted")
    ac_repo.update_status(reset, "rejected")

    uat_repo.persist_generated(
        chat_session.id,
        {
            login.id: [_case(login.id, "Valid login"), _case(login.id, "Wrong password")],
            logout.id: [_case(logout.id, "Logout clears session")],
        },
    )
    by_title = {case.title: case for case in uat_repo.list_for_session(chat_session.id)}
    uat_repo.update_status(by_title["Valid login"], "accepted")
    uat_repo.update_status(by_title["Wrong password"], "rejected")

    return ExportService(sessions, ac_repo, uat_repo), chat_session.id


def test_only_approved_acs_are_exported_with_their_approved_cases_nested(db_session) -> None:
    service, session_id = _seed(db_session)

    document = service.build(session_id).document

    assert document.session_name == "Auth stories"
    assert [ac.title for ac in document.acceptance_criteria] == ["Login", "Logout"]
    login, logout = document.acceptance_criteria
    assert [case.title for case in login.uat_cases] == ["Valid login"]
    assert login.uat_cases[0].description == "Valid login steps"
    assert login.uat_cases[0].source_agent == "agent-b"
    assert login.source_agent == "agent-a"
    assert logout.uat_cases == []


def test_warnings_name_what_the_export_leaves_out(db_session) -> None:
    service, session_id = _seed(db_session)

    warnings = service.build(session_id).warnings

    # Signup is pending and Reset is rejected; neither is approved.
    assert warnings.unapproved_ac_count == 2
    assert warnings.approved_ac_without_uat == ["Logout"]


def test_an_empty_session_exports_nothing_and_warns_about_nothing(db_session) -> None:
    sessions = SessionRepository(db_session)
    chat_session = sessions.create("Empty")
    service = ExportService(
        sessions, AcceptanceCriteriaRepository(db_session), UatCaseRepository(db_session)
    )

    result = service.build(chat_session.id)

    assert result.document.acceptance_criteria == []
    assert result.warnings.unapproved_ac_count == 0
    assert result.warnings.approved_ac_without_uat == []


def test_a_missing_session_returns_none(db_session) -> None:
    service = ExportService(
        SessionRepository(db_session),
        AcceptanceCriteriaRepository(db_session),
        UatCaseRepository(db_session),
    )

    assert service.build(999) is None
