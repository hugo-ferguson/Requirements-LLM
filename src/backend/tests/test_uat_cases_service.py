import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
)
from app.models_uat_cases import UatRegenerateSelectedRequest
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.agents import GenerationError
from app.services.scoring import CandidateScore
from app.services.uat_cases import GeneratedUatCase, GeneratedUatCases, UatCaseService


@pytest.fixture(name="db_session")
def db_session_fixture():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _seed_accepted_ac(db_session: Session) -> tuple[int, int]:
    """Creates a session with one accepted AC record; returns (session_id, ac_id)."""
    sessions = SessionRepository(db_session)
    ac_repo = AcceptanceCriteriaRepository(db_session)

    chat_session = sessions.create("test session")
    seeded = AcceptanceCriterion(
        id=0,
        title="Login succeeds",
        given="a registered user",
        when="they submit valid credentials",
        then="they land on the dashboard",
        scores=AcceptanceCriterionScores(relevance=1, correctness=1, understandability=1, coverage=1),
        overall_score=1,
    )
    [record] = ac_repo.persist_batch(chat_session.id, [seeded])
    ac_repo.update_status(record, "accepted")
    return chat_session.id, record.id


_FAKE_SCORE = CandidateScore(relevance=3, correctness=3.5, understandability=4, coverage=4.5, overall=3.75)


def _fake_scorer(prompt, candidates, *, reference_answer="", settings=None):
    return [_FAKE_SCORE for _ in candidates]


def _service(
    db_session: Session, *, generation_agent=None, regen_agent=None, scorer=None
) -> UatCaseService:
    return UatCaseService(
        SessionRepository(db_session),
        UatCaseRepository(db_session),
        AcceptanceCriteriaRepository(db_session),
        generation_agent=generation_agent,
        regen_agent=regen_agent,
        scorer=scorer or _fake_scorer,
    )


def _tool_response(info: AgentInfo, **fields) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=fields)]
    )


def test_generate_persists_real_cases_from_the_model_scored_by_the_scorer(
    db_session: Session,
) -> None:
    session_id, ac_id = _seed_accepted_ac(db_session)

    def write_cases(messages, info):
        return _tool_response(
            info,
            cases=[
                {"title": "Happy path", "description": "Valid credentials log the user in"},
                {"title": "Wrong password", "description": "An error is shown, no login occurs"},
            ],
        )

    service = _service(
        db_session,
        generation_agent=Agent(
            FunctionModel(write_cases), output_type=GeneratedUatCases, system_prompt="test"
        ),
    )

    result = service.generate(session_id)

    assert result is not None
    [group] = result.groups
    assert group.ac.id == ac_id
    assert len(group.uat_cases) == 2
    titles = {c.title for c in group.uat_cases}
    assert titles == {"Happy path", "Wrong password"}
    for case in group.uat_cases:
        assert case.scores.relevance == 3
        assert case.overall_score == 3.75
        assert case.status == "pending"


def test_generate_raises_generation_error_and_persists_nothing_when_the_model_fails(
    db_session: Session,
) -> None:
    session_id, _ = _seed_accepted_ac(db_session)

    def explode(messages, info):
        raise RuntimeError("upstream is down")

    service = _service(
        db_session,
        generation_agent=Agent(FunctionModel(explode), output_type=GeneratedUatCases, system_prompt="test"),
    )

    with pytest.raises(GenerationError):
        service.generate(session_id)

    listed = service.list_items(session_id)
    assert listed.groups == []


def test_regenerate_selected_returns_exactly_one_candidate_reflecting_the_model_and_scorer(
    db_session: Session,
) -> None:
    session_id, ac_id = _seed_accepted_ac(db_session)

    def write_cases(messages, info):
        return _tool_response(
            info,
            cases=[
                {"title": "Original", "description": "Original description"},
                {"title": "Second", "description": "Second description"},
            ],
        )

    seed_service = _service(
        db_session,
        generation_agent=Agent(
            FunctionModel(write_cases), output_type=GeneratedUatCases, system_prompt="test"
        ),
    )
    seed_service.generate(session_id)
    [group] = seed_service.list_items(session_id).groups
    target_id = group.uat_cases[0].id

    def refine(messages, info):
        return _tool_response(info, title="Refined title", description="Refined description")

    service = _service(
        db_session,
        regen_agent=Agent(FunctionModel(refine), output_type=GeneratedUatCase, system_prompt="test"),
    )

    response = service.regenerate_selected(
        session_id,
        target_id,
        UatRegenerateSelectedRequest(
            messages=[ConversationMessage(role="user", text="add more detail")]
        ),
    )

    assert response is not None
    assert len(response.candidates) == 1
    candidate = response.candidates[0]
    assert candidate.title == "Refined title"
    assert candidate.description == "Refined description"
    assert candidate.ac_id == ac_id
    assert candidate.scores.relevance == 3
    assert response.reply.role == "assistant"


def test_regenerate_selected_returns_none_for_a_missing_uat_case(db_session: Session) -> None:
    session_id, _ = _seed_accepted_ac(db_session)

    def explode(messages, info):
        raise AssertionError("should not be called for a missing UAT case")

    service = _service(
        db_session,
        regen_agent=Agent(FunctionModel(explode), output_type=GeneratedUatCase, system_prompt="test"),
    )

    result = service.regenerate_selected(
        session_id, 9999, UatRegenerateSelectedRequest(messages=[])
    )

    assert result is None
