"""UAT cases from every model, with alternatives that can be swapped in."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.generation import orchestrator
from app.generation.models import NumberedUatCase, NumberedUatCaseSet
from app.generation.orchestrator import run_uat_ensemble
from app.llm_config import GenerationAgentConfig
from app.models_conversation import AcceptanceCriterion, AcceptanceCriterionScores
from app.models_uat_cases import (
    UatApplyApprovedRequest,
    UatCase,
    UatCaseAlternative,
    UatCaseScores,
    UatCaseTextUpdate,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.uat_cases import UatCaseService


def _agent(agent_id: str) -> GenerationAgentConfig:
    """Return an offline test agent's config."""
    return GenerationAgentConfig(
        id=agent_id, provider="test", model="test", temperature=0.0, output_mode="tool"
    )


async def test_every_follower_answer_lands_on_the_anchors_case_titles():
    results, groups = await run_uat_ensemble(
        "Title: Log in", agents=[_agent("anchor"), _agent("follower")]
    )

    assert all(result.ok for result in results)
    assert [group.title for group in groups] == [
        "Log in with valid credentials",
        "Reject a wrong password",
    ]
    for group in groups:
        assert [candidate.agent_id for candidate in group.candidates] == [
            "anchor",
            "follower",
        ]
        assert group.candidates[1].case.title == group.title


async def test_the_next_agent_anchors_when_the_first_one_fails(monkeypatch):
    real_uat_agent = orchestrator._uat_agent

    def build(config, output_type):
        if config.id == "broken":
            raise RuntimeError("down")
        return real_uat_agent(config, output_type)

    monkeypatch.setattr(orchestrator, "_uat_agent", build)

    results, groups = await run_uat_ensemble(
        "Title: Log in", agents=[_agent("broken"), _agent("healthy")]
    )

    assert [result.agent_id for result in results] == ["broken", "healthy"]
    assert not results[0].ok
    assert groups
    assert all(group.candidates[0].agent_id == "healthy" for group in groups)


async def test_every_agent_failing_yields_no_groups(monkeypatch):
    def build(config, output_type):
        raise RuntimeError("down")

    monkeypatch.setattr(orchestrator, "_uat_agent", build)

    results, groups = await run_uat_ensemble(
        "Title: Log in", agents=[_agent("a"), _agent("b")]
    )

    assert groups == []
    assert [result.ok for result in results] == [False, False]


def test_unknown_and_repeated_case_numbers_are_dropped():
    output = NumberedUatCaseSet(
        cases=[
            NumberedUatCase(title_number=number, description=f"description {number}")
            for number in (2, 5, 2, 1)
        ]
    )

    cases = orchestrator._attach_uat_titles("follower", output, ["First", "Second"])

    assert [(case.title, case.description) for case in cases] == [
        ("Second", "description 2"),
        ("First", "description 1"),
    ]


@pytest.fixture(name="db_session")
def db_session_fixture():
    """Yield a session on a fresh in-memory SQLite database."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _scores(value: float) -> UatCaseScores:
    """Return rubric scores that are all `value`."""
    return UatCaseScores(
        relevance=value, correctness=value, understandability=value, coverage=value
    )


def _case(ac_id: int, index: int) -> UatCase:
    """Return case `index`, written by agent-a with agent-b's as an alternative.

    Agent-a's version scores 4 and agent-b's scores 3.
    """
    return UatCase(
        id=-1,
        ac_id=ac_id,
        title=f"Case {index}",
        description=f"a description {index}",
        scores=_scores(4),
        overall_score=4,
        source_agent="agent-a",
        alternatives=[
            UatCaseAlternative(
                description=f"b description {index}",
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


def _seed(db_session: Session, count: int = 3) -> tuple[UatCaseService, int, list[int]]:
    """Store one AC with `count` cases; return the service, session and case IDs."""
    chat_session = SessionRepository(db_session).create("test session")
    session_id = _id(chat_session.id)
    [ac] = AcceptanceCriteriaRepository(db_session).persist_batch(
        session_id,
        [
            AcceptanceCriterion(
                id=0,
                title="Log in",
                given="g",
                when="w",
                then="t",
                scores=AcceptanceCriterionScores(
                    relevance=1, correctness=1, understandability=1, coverage=1
                ),
                overall_score=1,
            )
        ],
    )
    ac_id = _id(ac.id)
    uat_case_repository = UatCaseRepository(db_session)
    uat_case_repository.persist_batch(
        session_id, {ac_id: [_case(ac_id, index) for index in range(count)]}
    )
    service = UatCaseService(
        SessionRepository(db_session),
        uat_case_repository,
        AcceptanceCriteriaRepository(db_session),
    )
    records = uat_case_repository.list_for_ac(ac_id)
    return service, session_id, [_id(record.id) for record in records]


def _cases(service: UatCaseService, session_id: int) -> list[UatCase]:
    """Return the cases of the session's only AC."""
    result = service.list_for_session(session_id)
    assert result is not None
    [group] = result.groups
    return group.uat_cases


def _select(
    service: UatCaseService, session_id: int, uat_id: int, candidate_id: int | None
) -> UatCase:
    """Select a candidate, which must succeed."""
    case = service.select_alternative(session_id, uat_id, _id(candidate_id))
    assert case is not None
    return case


def test_list_includes_alternatives_and_source_agent(db_session: Session) -> None:
    service, session_id, _ = _seed(db_session)

    for index, case in enumerate(_cases(service, session_id)):
        assert case.source_agent == "agent-a"
        [alternative] = case.alternatives
        assert (
            alternative.source_agent,
            alternative.description,
            alternative.overall_score,
        ) == ("agent-b", f"b description {index}", 3)


def test_select_alternative_swaps_and_swapping_back_restores(
    db_session: Session,
) -> None:
    service, session_id, uat_ids = _seed(db_session)
    record = service.uat_case_repository.get(session_id, uat_ids[1])
    assert record is not None
    service.uat_case_repository.update_status(record, "accepted")
    [alternative] = _cases(service, session_id)[1].alternatives

    swapped = _select(service, session_id, uat_ids[1], alternative.candidate_id)

    assert (swapped.description, swapped.overall_score, swapped.source_agent) == (
        "b description 1",
        3,
        "agent-b",
    )
    assert (swapped.title, swapped.status) == ("Case 1", "accepted")
    [back] = swapped.alternatives
    assert back.description == "a description 1"
    assert _cases(service, session_id)[0].description == "a description 0"

    restored = _select(service, session_id, uat_ids[1], back.candidate_id)
    assert (restored.description, restored.source_agent) == (
        "a description 1",
        "agent-a",
    )


def test_manual_edit_survives_as_the_alternative(db_session: Session) -> None:
    service, session_id, uat_ids = _seed(db_session)
    service.update_text(
        session_id, uat_ids[0], UatCaseTextUpdate(title="Case 0", description="edited")
    )
    [alternative] = _cases(service, session_id)[0].alternatives

    swapped = _select(service, session_id, uat_ids[0], alternative.candidate_id)

    assert swapped.alternatives[0].description == "edited"


def test_select_alternative_rejects_a_candidate_from_another_case(
    db_session: Session,
) -> None:
    service, session_id, uat_ids = _seed(db_session)
    other_case_alternative = _cases(service, session_id)[1].alternatives[0]

    assert (
        service.select_alternative(
            session_id, uat_ids[0], _id(other_case_alternative.candidate_id)
        )
        is None
    )
    assert service.select_alternative(session_id, uat_ids[0], 999_999) is None
    assert _cases(service, session_id)[0].description == "a description 0"


def test_apply_approved_keeps_other_cases_alternatives_on_the_right_case(
    db_session: Session,
) -> None:
    service, session_id, uat_ids = _seed(db_session)
    replacement = UatCase(
        id=-1,
        ac_id=0,
        title="Replacement",
        description="new",
        scores=_scores(5),
        overall_score=5,
    )

    group = service.apply_approved(
        session_id,
        uat_ids[0],
        UatApplyApprovedRequest(
            candidates=[replacement, replacement.model_copy(update={"title": "Extra"})]
        ),
    )

    assert group is not None
    cases = group.uat_cases
    assert [case.title for case in cases] == [
        "Replacement",
        "Extra",
        "Case 1",
        "Case 2",
    ]
    assert cases[0].alternatives == [] and cases[1].alternatives == []
    assert cases[2].alternatives[0].description == "b description 1"
    assert cases[3].alternatives[0].description == "b description 2"


def test_generate_and_select_alternative_routes(client: TestClient) -> None:
    session = client.post("/sessions", json={}).json()
    client.post(
        f"/sessions/{session['id']}/messages",
        json={
            "text": "As a user, I want to log in with my email and password.",
            "attachments": [],
        },
    )
    [ac, *_] = client.post(f"/sessions/{session['id']}/generate").json()[
        "acceptance_criteria"
    ]
    client.patch(
        f"/sessions/{session['id']}/acceptance-criteria/{ac['id']}/status",
        json={"status": "accepted"},
    )

    groups = client.post(f"/sessions/{session['id']}/uat-cases/generate").json()[
        "groups"
    ]
    target = groups[0]["uat_cases"][0]
    [alternative] = target["alternatives"]

    response = client.post(
        f"/sessions/{session['id']}/uat-cases/{target['id']}"
        f"/alternatives/{alternative['candidate_id']}/select"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_agent"] == alternative["source_agent"]
    assert body["description"] == alternative["description"]
    assert body["alternatives"][0]["source_agent"] == target["source_agent"]

    missing = client.post(
        f"/sessions/{session['id']}/uat-cases/{target['id']}/alternatives/999999/select"
    )
    assert missing.status_code == 404
