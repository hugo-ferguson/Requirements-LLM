"""Showing other models' versions of an AC, and swapping one in."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.models_acceptance_criteria import (
    AcceptanceCriterionTextUpdate,
    ApplyApprovedRequest,
    SelectAlternativeResponse,
)
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionAlternative,
    AcceptanceCriterionScores,
)
from app.models_uat_cases import UatCase, UatCaseScores
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.acceptance_criteria import AcceptanceCriteriaService


@pytest.fixture(name="db_session")
def db_session_fixture():
    """Yield a session on a fresh in-memory SQLite database."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _scores(value: float) -> AcceptanceCriterionScores:
    """Return rubric scores that are all `value`."""
    return AcceptanceCriterionScores(
        relevance=value, correctness=value, understandability=value, coverage=value
    )


def _criterion(index: int) -> AcceptanceCriterion:
    """Return AC `index`, written by agent-a with agent-b's as an alternative.

    Agent-a's version scores 4 and agent-b's scores 3.
    """
    return AcceptanceCriterion(
        id=0,
        title=f"Title {index}",
        given=f"a given {index}",
        when=f"a when {index}",
        then=f"a then {index}",
        scores=_scores(4),
        overall_score=4,
        source_agent="agent-a",
        alternatives=[
            AcceptanceCriterionAlternative(
                given=f"b given {index}",
                when=f"b when {index}",
                then=f"b then {index}",
                scores=_scores(3),
                overall_score=3,
                source_agent="agent-b",
            )
        ],
    )


def _id(value: int | None) -> int:
    """Return a stored row's ID, which the database must have assigned."""
    assert value is not None
    return value


def _criteria(
    service: AcceptanceCriteriaService, session_id: int
) -> list[AcceptanceCriterion]:
    """Return the session's ACs, which must exist."""
    criteria = service.list_for_session(session_id)
    assert criteria is not None
    return criteria


def _select(
    service: AcceptanceCriteriaService,
    session_id: int,
    ac_id: int,
    candidate_id: int | None,
) -> SelectAlternativeResponse:
    """Select a candidate, which must succeed."""
    result = service.select_alternative(session_id, ac_id, _id(candidate_id))
    assert result is not None
    return result


def _seed(
    db_session: Session, count: int = 3
) -> tuple[AcceptanceCriteriaService, int, list[int]]:
    """Store `count` ACs; return the service, session ID and AC IDs."""
    chat_session = SessionRepository(db_session).create("test session")
    records = AcceptanceCriteriaRepository(db_session).persist_batch(
        _id(chat_session.id), [_criterion(index) for index in range(count)]
    )
    service = AcceptanceCriteriaService(
        SessionRepository(db_session),
        AcceptanceCriteriaRepository(db_session),
        MessageRepository(db_session),
        UatCaseRepository(db_session),
    )
    return service, _id(chat_session.id), [_id(record.id) for record in records]


def test_list_includes_alternatives_and_source_agent(db_session: Session) -> None:
    service, session_id, _ = _seed(db_session)

    criteria = _criteria(service, session_id)

    assert [criterion.source_agent for criterion in criteria] == ["agent-a"] * 3
    for index, criterion in enumerate(criteria):
        [alternative] = criterion.alternatives
        assert alternative.source_agent == "agent-b"
        assert alternative.given == f"b given {index}"
        assert alternative.overall_score == 3
        assert alternative.candidate_id is not None


def test_select_alternative_swaps_text_and_scores_but_keeps_title_and_status(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    record = service.ac_repository.get(session_id, ac_ids[1])
    assert record is not None
    service.ac_repository.update_status(record, "accepted")
    [alternative] = _criteria(service, session_id)[1].alternatives

    result = _select(service, session_id, ac_ids[1], alternative.candidate_id)

    swapped = result.acceptance_criterion
    assert (swapped.given, swapped.when, swapped.then) == (
        "b given 1",
        "b when 1",
        "b then 1",
    )
    assert swapped.overall_score == 3
    assert swapped.scores == _scores(3)
    assert swapped.title == "Title 1"
    assert swapped.status == "accepted"
    assert swapped.source_agent == "agent-b"
    [demoted] = swapped.alternatives
    assert demoted.source_agent == "agent-a"
    assert demoted.given == "a given 1"
    assert result.uat_cases_affected == 0

    # Neighbours are untouched.
    criteria = _criteria(service, session_id)
    assert criteria[0].given == "a given 0"
    assert criteria[2].given == "a given 2"


def test_swapping_back_restores_the_original(db_session: Session) -> None:
    service, session_id, ac_ids = _seed(db_session)
    [alternative] = _criteria(service, session_id)[0].alternatives
    swapped = _select(service, session_id, ac_ids[0], alternative.candidate_id)
    [back] = swapped.acceptance_criterion.alternatives

    restored = _select(service, session_id, ac_ids[0], back.candidate_id)

    criterion = restored.acceptance_criterion
    assert (criterion.given, criterion.overall_score, criterion.source_agent) == (
        "a given 0",
        4,
        "agent-a",
    )
    assert criterion.alternatives[0].given == "b given 0"


def test_manual_edit_survives_as_the_alternative(db_session: Session) -> None:
    service, session_id, ac_ids = _seed(db_session)
    service.update_text(
        session_id,
        ac_ids[0],
        AcceptanceCriterionTextUpdate(
            title="Title 0",
            given="edited given",
            when="edited when",
            then="edited then",
        ),
    )
    [alternative] = _criteria(service, session_id)[0].alternatives

    result = _select(service, session_id, ac_ids[0], alternative.candidate_id)

    [demoted] = result.acceptance_criterion.alternatives
    assert (demoted.given, demoted.when, demoted.then) == (
        "edited given",
        "edited when",
        "edited then",
    )


def test_selecting_the_current_winner_is_a_no_op(db_session: Session) -> None:
    service, session_id, ac_ids = _seed(db_session)
    [winner] = [
        candidate
        for candidate in service.ac_repository.candidates_for_session(session_id)
        if candidate.criterion_position == 0 and candidate.is_winner
    ]

    result = _select(service, session_id, ac_ids[0], winner.id)

    assert result.acceptance_criterion.given == "a given 0"
    assert len(result.acceptance_criterion.alternatives) == 1


def test_select_alternative_rejects_a_candidate_from_another_ac_or_session(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    other_ac_alternative = _criteria(service, session_id)[1].alternatives[0]
    _, other_session_id, _ = _seed(db_session)
    other_session_alternative = _criteria(service, other_session_id)[0].alternatives[0]

    assert (
        service.select_alternative(
            session_id, ac_ids[0], _id(other_ac_alternative.candidate_id)
        )
        is None
    )
    assert (
        service.select_alternative(
            session_id, ac_ids[0], _id(other_session_alternative.candidate_id)
        )
        is None
    )
    assert service.select_alternative(session_id, ac_ids[0], 999_999) is None
    assert (
        service.select_alternative(
            session_id, 999_999, _id(other_ac_alternative.candidate_id)
        )
        is None
    )
    # Nothing changed.
    assert _criteria(service, session_id)[0].given == "a given 0"


def test_select_alternative_reports_uat_cases_written_against_the_old_version(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    UatCaseRepository(db_session).persist_batch(
        session_id, {ac_ids[0]: [_uat_case(ac_ids[0]), _uat_case(ac_ids[0])]}
    )
    [alternative] = _criteria(service, session_id)[0].alternatives

    result = _select(service, session_id, ac_ids[0], alternative.candidate_id)

    assert result.uat_cases_affected == 2


def test_every_ac_reports_how_many_uat_cases_it_has(db_session: Session) -> None:
    """The frontend warns before changing an AC whose count is above zero."""
    service, session_id, ac_ids = _seed(db_session)
    UatCaseRepository(db_session).persist_batch(
        session_id,
        {
            ac_ids[0]: [_uat_case(ac_ids[0]), _uat_case(ac_ids[0])],
            ac_ids[2]: [_uat_case(ac_ids[2])],
        },
    )

    criteria = _criteria(service, session_id)
    assert [criterion.uat_case_count for criterion in criteria] == [2, 0, 1]

    # Single-AC responses include the count too.
    [alternative] = criteria[0].alternatives
    assert (
        _select(
            service, session_id, ac_ids[0], alternative.candidate_id
        ).acceptance_criterion.uat_case_count
        == 2
    )
    edited = service.update_text(
        session_id,
        ac_ids[2],
        AcceptanceCriterionTextUpdate(title="Edited", given="g", when="w", then="t"),
    )
    assert edited.uat_case_count == 1


def test_apply_approved_reports_the_replaced_acs_uat_cases_as_gone(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    UatCaseRepository(db_session).persist_batch(
        session_id,
        {ac_ids[0]: [_uat_case(ac_ids[0])], ac_ids[1]: [_uat_case(ac_ids[1])]},
    )
    replacement = AcceptanceCriterion(
        id=-1,
        title="Replacement",
        given="g",
        when="w",
        then="t",
        scores=_scores(5),
        overall_score=5,
    )

    criteria = service.apply_approved(
        session_id, ac_ids[0], ApplyApprovedRequest(candidates=[replacement])
    )

    assert criteria is not None
    assert [criterion.uat_case_count for criterion in criteria] == [0, 1, 0]


def test_apply_approved_keeps_other_acs_alternatives_on_the_right_ac(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    replacement = AcceptanceCriterion(
        id=-1,
        title="Replacement",
        given="new given",
        when="new when",
        then="new then",
        scores=_scores(5),
        overall_score=5,
    )
    extra = replacement.model_copy(update={"title": "Extra", "given": "extra given"})

    # Both replacements take the first AC's slot, pushing the rest down.
    criteria = service.apply_approved(
        session_id, ac_ids[0], ApplyApprovedRequest(candidates=[replacement, extra])
    )
    assert criteria is not None

    assert [criterion.title for criterion in criteria] == [
        "Replacement",
        "Extra",
        "Title 1",
        "Title 2",
    ]
    assert criteria[0].alternatives == []
    assert criteria[1].alternatives == []
    assert criteria[2].alternatives[0].given == "b given 1"
    assert criteria[3].alternatives[0].given == "b given 2"
    # The replaced AC's candidates are deleted with it.
    positions = {
        candidate.criterion_position
        for candidate in service.ac_repository.candidates_for_session(session_id)
    }
    assert positions == {2, 3}


def test_select_alternative_route(client: TestClient) -> None:
    session = client.post("/sessions", json={}).json()
    client.post(
        f"/sessions/{session['id']}/messages",
        json={
            "text": "As a user, I want to log in with my email and password.",
            "attachments": [],
        },
    )
    generated = client.post(f"/sessions/{session['id']}/generate").json()[
        "acceptance_criteria"
    ]
    # Both test agents answer each title, so ACs have alternatives.
    target = next(criterion for criterion in generated if criterion["alternatives"])
    alternative = target["alternatives"][0]

    response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}"
        f"/alternatives/{alternative['candidate_id']}/select"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["acceptance_criterion"]["source_agent"] == alternative["source_agent"]
    assert (
        body["acceptance_criterion"]["alternatives"][0]["source_agent"]
        == target["source_agent"]
    )
    assert body["uat_cases_affected"] == 0

    missing = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}/alternatives/999999/select"
    )
    assert missing.status_code == 404


def _uat_case(ac_id: int) -> UatCase:
    """Return a pending UAT case for the AC."""
    return UatCase(
        id=-1,
        ac_id=ac_id,
        title="case",
        description="description",
        scores=UatCaseScores(
            relevance=1, correctness=1, understandability=1, coverage=1
        ),
        overall_score=1,
        status="pending",
    )
