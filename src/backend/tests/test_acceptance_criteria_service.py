import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.models_acceptance_criteria import RegenerateSelectedRequest
from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
)
from app.repositories.acceptance_criteria import AcceptanceCriteriaRepository
from app.repositories.messages import MessageRepository
from app.repositories.sessions import SessionRepository
from app.repositories.uat_cases import UatCaseRepository
from app.services.acceptance_criteria import AcceptanceCriteriaService
from app.services.agents import GenerationError
from app.services.conversation import GeneratedCriterion
from app.services.scoring import CandidateScore


@pytest.fixture(name="db_session")
def db_session_fixture():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _seed_ac(db_session: Session) -> tuple[int, int]:
    """Creates a session with one AC record; returns (session_id, ac_id)."""
    sessions = SessionRepository(db_session)
    ac_repo = AcceptanceCriteriaRepository(db_session)

    chat_session = sessions.create("test session")
    seeded = AcceptanceCriterion(
        id=0,
        title="Original title",
        given="original given",
        when="original when",
        then="original then",
        scores=AcceptanceCriterionScores(relevance=1, correctness=1, understandability=1, coverage=1),
        overall_score=1,
    )
    [record] = ac_repo.persist_batch(chat_session.id, [seeded])
    return chat_session.id, record.id


def _service(db_session: Session, agent: Agent, scorer=None) -> AcceptanceCriteriaService:
    return AcceptanceCriteriaService(
        SessionRepository(db_session),
        AcceptanceCriteriaRepository(db_session),
        MessageRepository(db_session),
        UatCaseRepository(db_session),
        regen_agent=agent,
        scorer=scorer
        or (lambda prompt, candidates, *, reference_answer="", settings=None: [
            CandidateScore(relevance=3.5, correctness=4, understandability=4.5, coverage=3, overall=3.75)
            for _ in candidates
        ]),
    )


def _tool_response(info: AgentInfo, **fields) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=fields)]
    )


def test_regenerate_selected_returns_exactly_one_candidate_reflecting_the_model_and_scorer(
    db_session: Session,
) -> None:
    session_id, ac_id = _seed_ac(db_session)

    def refine(messages, info):
        return _tool_response(
            info,
            title="Refined title",
            given="refined given",
            when="refined when",
            then="refined then",
        )

    service = _service(db_session, Agent(FunctionModel(refine), output_type=GeneratedCriterion, system_prompt="test"))

    response = service.regenerate_selected(
        session_id,
        ac_id,
        RegenerateSelectedRequest(
            messages=[ConversationMessage(role="user", text="please tighten this up")]
        ),
    )

    assert response is not None
    assert len(response.candidates) == 1
    candidate = response.candidates[0]
    assert candidate.title == "Refined title"
    assert candidate.given == "refined given"
    assert candidate.scores.relevance == 3.5
    assert candidate.overall_score == 3.75
    assert candidate.status == "pending"
    assert response.reply.role == "assistant"


def test_regenerate_selected_returns_none_for_a_missing_ac(db_session: Session) -> None:
    session_id, _ = _seed_ac(db_session)

    def refine(messages, info):
        raise AssertionError("should not be called for a missing AC")

    service = _service(db_session, Agent(FunctionModel(refine), output_type=GeneratedCriterion, system_prompt="test"))

    result = service.regenerate_selected(session_id, 9999, RegenerateSelectedRequest(messages=[]))

    assert result is None


def test_regenerate_selected_raises_generation_error_when_the_model_fails(
    db_session: Session,
) -> None:
    session_id, ac_id = _seed_ac(db_session)

    def explode(messages, info):
        raise RuntimeError("upstream is down")

    service = _service(db_session, Agent(FunctionModel(explode), output_type=GeneratedCriterion, system_prompt="test"))

    with pytest.raises(GenerationError) as caught:
        service.regenerate_selected(session_id, ac_id, RegenerateSelectedRequest(messages=[]))

    assert "upstream is down" in str(caught.value)
