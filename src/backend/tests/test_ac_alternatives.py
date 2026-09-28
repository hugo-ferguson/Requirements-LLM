"""Showing other models' versions of an AC, and swapping one in."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.models_acceptance_criteria import AcceptanceCriterionTextUpdate, ApplyApprovedRequest
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
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _scores(value: float) -> AcceptanceCriterionScores:
    return AcceptanceCriterionScores(
        relevance=value, correctness=value, understandability=value, coverage=value
    )


def _criterion(n: int) -> AcceptanceCriterion:
    """AC `n`, written by agent-a (score 4), with agent-b's version as an alternative (score 3)."""
    return AcceptanceCriterion(
        id=0,
        title=f"Title {n}",
        given=f"a given {n}",
        when=f"a when {n}",
        then=f"a then {n}",
        scores=_scores(4),
        overall_score=4,
        source_agent="agent-a",
        alternatives=[
            AcceptanceCriterionAlternative(
                given=f"b given {n}",
                when=f"b when {n}",
                then=f"b then {n}",
                scores=_scores(3),
                overall_score=3,
                source_agent="agent-b",
            )
        ],
    )


def _seed(db_session: Session, count: int = 3) -> tuple[AcceptanceCriteriaService, int, list[int]]:
    chat_session = SessionRepository(db_session).create("test session")
    records = AcceptanceCriteriaRepository(db_session).persist_batch(
        chat_session.id, [_criterion(n) for n in range(count)]
    )
    service = AcceptanceCriteriaService(
        SessionRepository(db_session),
        AcceptanceCriteriaRepository(db_session),
        MessageRepository(db_session),
        UatCaseRepository(db_session),
    )
    return service, chat_session.id, [r.id for r in records]


def test_list_includes_alternatives_and_source_agent(db_session: Session) -> None:
    service, session_id, _ = _seed(db_session)

    items = service.list_items(session_id)

    assert [item.source_agent for item in items] == ["agent-a"] * 3
    for n, item in enumerate(items):
        [alt] = item.alternatives
        assert alt.source_agent == "agent-b"
        assert alt.given == f"b given {n}"
        assert alt.overall_score == 3
        assert alt.candidate_id is not None


def test_select_alternative_swaps_text_and_scores_but_keeps_title_and_status(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    service.ac.update_status(service.ac.get(session_id, ac_ids[1]), "accepted")
    [alt] = service.list_items(session_id)[1].alternatives

    result = service.select_alternative(session_id, ac_ids[1], alt.candidate_id)

    swapped = result.acceptance_criterion
    assert (swapped.given, swapped.when, swapped.then) == ("b given 1", "b when 1", "b then 1")
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
    items = service.list_items(session_id)
    assert items[0].given == "a given 0"
    assert items[2].given == "a given 2"


def test_swapping_back_restores_the_original(db_session: Session) -> None:
    service, session_id, ac_ids = _seed(db_session)
    [alt] = service.list_items(session_id)[0].alternatives
    swapped = service.select_alternative(session_id, ac_ids[0], alt.candidate_id)
    [back] = swapped.acceptance_criterion.alternatives

    restored = service.select_alternative(session_id, ac_ids[0], back.candidate_id)

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
            title="Title 0", given="edited given", when="edited when", then="edited then"
        ),
    )
    [alt] = service.list_items(session_id)[0].alternatives

    result = service.select_alternative(session_id, ac_ids[0], alt.candidate_id)

    [demoted] = result.acceptance_criterion.alternatives
    assert (demoted.given, demoted.when, demoted.then) == (
        "edited given",
        "edited when",
        "edited then",
    )


def test_selecting_the_current_winner_is_a_no_op(db_session: Session) -> None:
    service, session_id, ac_ids = _seed(db_session)
    [winner] = [
        c for c in service.ac.candidates_for_session(session_id)
        if c.criterion_position == 0 and c.is_winner
    ]

    result = service.select_alternative(session_id, ac_ids[0], winner.id)

    assert result.acceptance_criterion.given == "a given 0"
    assert len(result.acceptance_criterion.alternatives) == 1


def test_select_alternative_rejects_a_candidate_from_another_ac_or_session(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    other_ac_alt = service.list_items(session_id)[1].alternatives[0]
    _, other_session_id, _ = _seed(db_session)
    other_session_alt = service.list_items(other_session_id)[0].alternatives[0]

    assert service.select_alternative(session_id, ac_ids[0], other_ac_alt.candidate_id) is None
    assert service.select_alternative(session_id, ac_ids[0], other_session_alt.candidate_id) is None
    assert service.select_alternative(session_id, ac_ids[0], 999_999) is None
    assert service.select_alternative(session_id, 999_999, other_ac_alt.candidate_id) is None
    # Nothing changed.
    assert service.list_items(session_id)[0].given == "a given 0"


def test_select_alternative_reports_uat_cases_written_against_the_old_version(
    db_session: Session,
) -> None:
    service, session_id, ac_ids = _seed(db_session)
    UatCaseRepository(db_session).persist_generated(
        session_id, {ac_ids[0]: [_uat_case(ac_ids[0]), _uat_case(ac_ids[0])]}
    )
    [alt] = service.list_items(session_id)[0].alternatives

    result = service.select_alternative(session_id, ac_ids[0], alt.candidate_id)

    assert result.uat_cases_affected == 2


def test_apply_approved_keeps_other_acs_alternatives_on_the_right_ac(db_session: Session) -> None:
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

    # Both replacements take the first AC's slot, shifting every later AC down.
    items = service.apply_approved(
        session_id, ac_ids[0], ApplyApprovedRequest(candidates=[replacement, extra])
    )

    assert [item.title for item in items] == ["Replacement", "Extra", "Title 1", "Title 2"]
    assert items[0].alternatives == []
    assert items[1].alternatives == []
    assert items[2].alternatives[0].given == "b given 1"
    assert items[3].alternatives[0].given == "b given 2"
    # The replaced AC's candidates are gone, not left behind.
    positions = {c.criterion_position for c in service.ac.candidates_for_session(session_id)}
    assert positions == {2, 3}


def test_select_alternative_route(client: TestClient) -> None:
    session = client.post("/sessions", json={}).json()
    client.post(
        f"/sessions/{session['id']}/messages",
        json={"text": "As a user, I want to log in with my email and password.", "attachments": []},
    )
    generated = client.post(f"/sessions/{session['id']}/generate").json()["acceptance_criteria"]
    # The two-agent test roster answers the shared titles twice.
    target = next(item for item in generated if item["alternatives"])
    alt = target["alternatives"][0]

    response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}"
        f"/alternatives/{alt['candidate_id']}/select"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["acceptance_criterion"]["source_agent"] == alt["source_agent"]
    assert body["acceptance_criterion"]["alternatives"][0]["source_agent"] == target["source_agent"]
    assert body["uat_cases_affected"] == 0

    missing = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}/alternatives/999999/select"
    )
    assert missing.status_code == 404


def _uat_case(ac_id: int):
    return UatCase(
        id=-1,
        ac_id=ac_id,
        title="case",
        description="description",
        scores=UatCaseScores(relevance=1, correctness=1, understandability=1, coverage=1),
        overall_score=1,
        status="pending",
    )

