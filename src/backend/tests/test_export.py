"""The JSON export of approved ACs and their approved UAT cases."""

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
    """Yield a session on a fresh in-memory SQLite database."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _criterion(title: str) -> AcceptanceCriterion:
    """Return a pending AC written by agent-a."""
    scores = AcceptanceCriterionScores(
        relevance=4, correctness=4, understandability=4, coverage=4
    )
    return AcceptanceCriterion(
        id=0,
        title=title,
        given="g",
        when="w",
        then="t",
        scores=scores,
        overall_score=4,
        source_agent="agent-a",
    )


def _case(ac_id: int, title: str) -> UatCase:
    """Return a pending UAT case for the AC, written by agent-b."""
    scores = UatCaseScores(relevance=3, correctness=3, understandability=3, coverage=3)
    return UatCase(
        id=-1,
        ac_id=ac_id,
        title=title,
        description=f"{title} steps",
        scores=scores,
        overall_score=3,
        source_agent="agent-b",
    )


def _seed(db_session: Session) -> tuple[ExportService, int]:
    """Seed four ACs; return the export service and the session ID.

    Login is approved with one of its two cases approved, Logout is approved
    with no approved cases, Signup is pending and Reset is rejected.
    """
    session_repository = SessionRepository(db_session)
    ac_repository = AcceptanceCriteriaRepository(db_session)
    uat_case_repository = UatCaseRepository(db_session)

    chat_session = session_repository.create("Auth stories")
    titles = ("Login", "Logout", "Signup", "Reset")
    login, logout, _, reset = ac_repository.persist_batch(
        chat_session.id, [_criterion(title) for title in titles]
    )
    ac_repository.update_status(login, "accepted")
    ac_repository.update_status(logout, "accepted")
    ac_repository.update_status(reset, "rejected")

    uat_case_repository.persist_batch(
        chat_session.id,
        {
            login.id: [
                _case(login.id, "Valid login"),
                _case(login.id, "Wrong password"),
            ],
            logout.id: [_case(logout.id, "Logout clears session")],
        },
    )
    cases = uat_case_repository.list_for_session(chat_session.id)
    cases_by_title = {case.title: case for case in cases}
    uat_case_repository.update_status(cases_by_title["Valid login"], "accepted")
    uat_case_repository.update_status(cases_by_title["Wrong password"], "rejected")

    service = ExportService(session_repository, ac_repository, uat_case_repository)
    return service, chat_session.id


def test_only_approved_acs_are_exported_with_their_approved_cases_nested(
    db_session: Session,
) -> None:
    service, session_id = _seed(db_session)

    document = service.build(session_id).document

    assert document.session_name == "Auth stories"
    assert [criterion.title for criterion in document.acceptance_criteria] == [
        "Login",
        "Logout",
    ]
    login, logout = document.acceptance_criteria
    assert [case.title for case in login.uat_cases] == ["Valid login"]
    assert login.uat_cases[0].description == "Valid login steps"
    assert login.uat_cases[0].source_agent == "agent-b"
    assert login.source_agent == "agent-a"
    assert logout.uat_cases == []


def test_warnings_name_what_the_export_leaves_out(db_session: Session) -> None:
    service, session_id = _seed(db_session)

    warnings = service.build(session_id).warnings

    # Signup is pending and Reset is rejected; neither is approved.
    assert warnings.unapproved_ac_count == 2
    assert warnings.approved_ac_without_uat == ["Logout"]


def test_an_empty_session_exports_nothing_and_warns_about_nothing(
    db_session: Session,
) -> None:
    session_repository = SessionRepository(db_session)
    chat_session = session_repository.create("Empty")
    service = ExportService(
        session_repository,
        AcceptanceCriteriaRepository(db_session),
        UatCaseRepository(db_session),
    )

    result = service.build(chat_session.id)

    assert result.document.acceptance_criteria == []
    assert result.warnings.unapproved_ac_count == 0
    assert result.warnings.approved_ac_without_uat == []


def test_a_missing_session_returns_none(db_session: Session) -> None:
    service = ExportService(
        SessionRepository(db_session),
        AcceptanceCriteriaRepository(db_session),
        UatCaseRepository(db_session),
    )

    assert service.build(999) is None
