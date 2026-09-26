"""Offline tests for the ensemble generation layer.

Everything here runs without a database, Ollama or an API key: the roster is
pointed at PydanticAI's TestModel by the autouse fixture in conftest.py, and
the voting layer is faked in-process.

Tests are grouped by the backlog requirement they cover: R7 (valid Gherkin),
R8 (agents configured by file, not code), R10 (agents run concurrently) and
R11 (top-K retrieved context reaches the prompt).
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.generation import orchestrator
from app.generation.agents import AgentBuildError, build_agent
from app.generation.config import (
    GenerationAgentConfig,
    RosterConfigError,
    load_roster,
)
from app.generation.models import (
    AgentResult,
    Candidate,
    CandidateGroup,
    EnsembleResult,
    GeneratedCriteriaSet,
    GeneratedCriterion,
)
from app.generation.orchestrator import pool_criteria, run_ensemble
from app.generation.prompts import GenerationDeps, build_user_prompt
from app.services.scoring import CandidateScore
from app.models_conversation import ConversationAttachment, ConversationMessage
from app.services.generation import (
    _groups_to_wire_models,
    _to_wire_models,
    GenerationService,
    NoUserStoryError,
    compose_user_story,
)


def _test_agent(agent_id: str) -> GenerationAgentConfig:
    """A roster entry backed by TestModel, so no provider is ever contacted."""
    return GenerationAgentConfig(
        id=agent_id, provider="test", model="test", temperature=0.0
    )


def _criterion(number: int) -> GeneratedCriterion:
    """A distinguishable criterion; `title` is what the assertions key on."""
    return GeneratedCriterion(
        title=f"Title {number}",
        given=f"precondition {number}",
        when=f"action {number}",
        then=f"outcome {number}",
    )


def _settings(**overrides) -> Settings:
    """Settings with explicit values, so a stray .env can't change a result."""
    base = dict(
        rag_top_k=5,
        generation_max_criteria=8,
        generation_timeout_seconds=45.0,
        generation_enable_voting=False,
    )
    base.update(overrides)
    return Settings(**base)


class _FakeChunk:
    """Stands in for a vector-store row; only `.content` is read."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeIngest:
    """Records how it was called so the top-K contract can be asserted."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[_FakeChunk]:
        self.calls.append((query, k))
        return [_FakeChunk(text) for text in self._contents[:k]]


def _fake_scorer(overalls: list[float]):
    """A Scorer stub returning one CandidateScore per candidate, in order.

    app.services.scoring.score_candidates guarantees submission order, so the
    service zips its results against the candidates it sent. This stub holds
    to the same contract and asserts the count matches.
    """

    def scorer(prompt, candidates, *, reference_answer='', settings=None):
        assert len(candidates) == len(overalls), 'scorer got the wrong candidate count'
        return [
            CandidateScore(relevance=o, correctness=o, understandability=o,
                           coverage=o, overall=o)
            for o in overalls
        ]

    return scorer


@pytest.mark.asyncio
async def test_agents_run_concurrently_not_serially(monkeypatch) -> None:
    """Total time should track the slowest agent, not the sum of all agents."""
    from app.generation import orchestrator

    async def slow_run(config, deps, *, max_criteria, timeout_seconds):
        await asyncio.sleep(0.3)
        return orchestrator.AgentResult(
            agent_id=config.id, provider=config.provider, model=config.model
        )

    monkeypatch.setattr(orchestrator, "_run_one", slow_run)

    agents = [_test_agent(f"a{i}") for i in range(5)]
    started = time.perf_counter()
    await run_ensemble(GenerationDeps(user_story="story"), agents=agents)
    elapsed = time.perf_counter() - started

    # Serial would be ~1.5s; concurrent should be ~0.3s.
    assert elapsed < 0.9, f"agents appear to be running serially ({elapsed:.2f}s)"


def test_scores_land_on_their_own_criterion_and_rank_highest_first() -> None:
    """Each criterion keeps its own score, and the strongest sorts first.

    Scores arrive from app.services.scoring in submission order, so the pairing
    is positional. Ranking has to happen before persistence because the
    repository uses list order for `position` (backlog R14, R17).
    """
    criteria = [_criterion(1), _criterion(2), _criterion(3)]
    scores = _fake_scorer([2.0, 5.0, 1.0])(prompt='p', candidates=['a', 'b', 'c'])

    ranked = _to_wire_models(criteria, scores=scores)

    assert [item.title for item in ranked] == ['Title 2', 'Title 1', 'Title 3']
    assert [item.overall_score for item in ranked] == [5.0, 2.0, 1.0]
    assert ranked[0].scores.relevance == 5.0


def test_unscored_criteria_are_returned_when_voting_is_disabled() -> None:
    """Turning voting off must cost the scores, never the criteria."""
    criteria = [_criterion(1), _criterion(2)]

    wired = _to_wire_models(criteria, scores=None)

    assert len(wired) == 2
    assert all(item.overall_score == 0.0 for item in wired)
    assert all(item.scores.coverage == 0.0 for item in wired)

# ---------------------------------------------------------------------------
# R7 - generated criteria must be structurally valid Gherkin
#
# Validity is enforced by the output schema itself: PydanticAI validates every
# agent response against GeneratedCriteriaSet and retries on failure, so a
# malformed criterion never leaves an agent.
# ---------------------------------------------------------------------------


def test_criterion_requires_all_three_gherkin_clauses() -> None:
    with pytest.raises(ValidationError):
        GeneratedCriterion(
            title="Missing the then clause",
            given="a registered user is on the login page",
            when="they submit valid credentials",
        )


@pytest.mark.parametrize("clause", ["given", "when", "then"])
def test_criterion_rejects_an_empty_clause(clause: str) -> None:
    fields = {
        "title": "Successful login redirects home",
        "given": "a registered user is on the login page",
        "when": "they submit valid credentials",
        "then": "they are redirected to the home page",
    }
    fields[clause] = ""

    with pytest.raises(ValidationError):
        GeneratedCriterion(**fields)


def test_criteria_set_requires_at_least_one_criterion() -> None:
    """An agent returning an empty set is a failure, not an empty success."""
    with pytest.raises(ValidationError):
        GeneratedCriteriaSet(user_story_summary="A summary.", criteria=[])


def test_criteria_set_caps_the_number_of_criteria() -> None:
    with pytest.raises(ValidationError):
        GeneratedCriteriaSet(
            user_story_summary="A summary.",
            criteria=[_criterion(n) for n in range(11)],
        )


def test_as_sentence_renders_clauses_in_gherkin_order() -> None:
    """The linearised form the voting layer scores must stay Given/When/Then."""
    sentence = _criterion(1).as_sentence()

    assert sentence == "Given precondition 1, when action 1, then outcome 1."


async def test_generated_criteria_survive_a_full_ensemble_run() -> None:
    """End to end against TestModel: every pooled criterion is valid Gherkin."""
    ensemble = await run_ensemble(
        GenerationDeps(user_story="As a user I want to log in"),
        agents=[_test_agent("a"), _test_agent("b")],
    )

    pooled = pool_criteria(ensemble)

    assert pooled, "expected at least one candidate from the test roster"
    for _, criterion in pooled:
        assert criterion.given and criterion.when and criterion.then
        assert criterion.as_sentence().startswith("Given ")


# ---------------------------------------------------------------------------
# R8 - the agent pool is declared in a config file, not in code
# ---------------------------------------------------------------------------


def _write_roster(tmp_path, payload: dict):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_roster_loads_agents_from_file(tmp_path) -> None:
    path = _write_roster(
        tmp_path,
        {
            "agents": [
                {"id": "qwen", "provider": "ollama", "model": "qwen2.5:7b"},
                {"id": "llama", "provider": "ollama", "model": "llama3.1:8b"},
            ]
        },
    )

    roster = load_roster(path)

    assert [agent.id for agent in roster.agents] == ["qwen", "llama"]
    assert roster.agents[0].model == "qwen2.5:7b"


def test_roster_returns_only_enabled_agents(tmp_path) -> None:
    """Adding or retiring a model is a config edit, not a code change."""
    path = _write_roster(
        tmp_path,
        {
            "agents": [
                {"id": "qwen", "provider": "ollama", "model": "qwen2.5:7b"},
                {
                    "id": "gemini",
                    "provider": "google",
                    "model": "gemini-2.0-flash",
                    "enabled": False,
                },
            ]
        },
    )

    roster = load_roster(path)

    assert len(roster.agents) == 2
    assert [agent.id for agent in roster.enabled_agents()] == ["qwen"]


def test_roster_missing_file_names_the_path_and_the_fix(tmp_path) -> None:
    with pytest.raises(RosterConfigError) as error:
        load_roster(tmp_path / "absent.json")

    assert "generation_agents.example.json" in str(error.value)


def test_roster_rejects_malformed_json(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RosterConfigError, match="not valid JSON"):
        load_roster(path)


def test_roster_rejects_duplicate_agent_ids(tmp_path) -> None:
    """Ids attribute votes back to a generator, so they have to be unique."""
    path = _write_roster(
        tmp_path,
        {
            "agents": [
                {"id": "qwen", "provider": "ollama", "model": "qwen2.5:7b"},
                {"id": "qwen", "provider": "ollama", "model": "qwen2.5:14b"},
            ]
        },
    )

    with pytest.raises(RosterConfigError, match="Duplicate agent ids"):
        load_roster(path)


def test_roster_rejects_a_roster_with_nothing_enabled(tmp_path) -> None:
    path = _write_roster(
        tmp_path,
        {
            "agents": [
                {
                    "id": "qwen",
                    "provider": "ollama",
                    "model": "qwen2.5:7b",
                    "enabled": False,
                }
            ]
        },
    )

    with pytest.raises(RosterConfigError, match="No enabled agents"):
        load_roster(path)


def test_api_keys_are_read_from_the_environment_not_the_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file names the variable; the value is resolved at call time."""
    config = GenerationAgentConfig(
        id="gemini",
        provider="google",
        model="gemini-2.0-flash",
        api_key_env="A_TEST_KEY",
    )

    monkeypatch.delenv("A_TEST_KEY", raising=False)
    assert config.api_key() is None

    monkeypatch.setenv("A_TEST_KEY", "secret-value")
    assert config.api_key() == "secret-value"


def test_agent_without_its_key_fails_to_build_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GenerationAgentConfig(
        id="gemini",
        provider="google",
        model="gemini-2.0-flash",
        api_key_env="A_TEST_KEY",
    )
    monkeypatch.delenv("A_TEST_KEY", raising=False)

    with pytest.raises(AgentBuildError, match="A_TEST_KEY"):
        build_agent(config)


# ---------------------------------------------------------------------------
# R10 - agents run concurrently, and one bad agent degrades the ensemble
#       rather than failing the request
# ---------------------------------------------------------------------------


class _ExplodingAgent:
    async def run(self, prompt, deps=None):
        raise RuntimeError("upstream exploded")


class _HangingAgent:
    async def run(self, prompt, deps=None):
        await asyncio.sleep(30)


async def test_one_failing_agent_does_not_fail_the_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_build_agent = orchestrator.build_agent

    def build_or_explode(config):
        if config.id == "broken":
            return _ExplodingAgent()
        return real_build_agent(config)

    monkeypatch.setattr(orchestrator, "build_agent", build_or_explode)

    ensemble = await run_ensemble(
        GenerationDeps(user_story="story"),
        agents=[_test_agent("healthy"), _test_agent("broken")],
    )

    assert [result.agent_id for result in ensemble.successful] == ["healthy"]
    assert [result.agent_id for result in ensemble.failed] == ["broken"]
    assert "upstream exploded" in ensemble.failed[0].error
    assert pool_criteria(ensemble), "the healthy agent's work must survive"


async def test_a_hanging_agent_is_cut_off_at_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One slow model must not eat the whole R6 budget."""
    monkeypatch.setattr(orchestrator, "build_agent", lambda config: _HangingAgent())

    ensemble = await run_ensemble(
        GenerationDeps(user_story="story"),
        agents=[_test_agent("slow")],
        timeout_seconds=0.05,
    )

    assert ensemble.successful == []
    assert "Timed out" in ensemble.failed[0].error


async def test_every_agent_failing_is_reported_as_a_failed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service turns a total wipeout into a 502, not an empty success."""
    monkeypatch.setattr(orchestrator, "build_agent", lambda config: _ExplodingAgent())

    service = GenerationService(_settings())
    messages = [ConversationMessage(role="user", text="As a user I want to log in")]

    with pytest.raises(RuntimeError, match="Every generation agent failed"):
        await service.generate(messages)


def test_pooling_drops_criteria_two_agents_produced_identically() -> None:
    """Three models agreeing should appear once, not three times."""
    shared = _criterion(1)
    ensemble = EnsembleResult(
        prompt="p",
        results=[
            AgentResult(
                agent_id="a",
                provider="test",
                model="test",
                criteria=[shared, _criterion(2)],
            ),
            AgentResult(
                agent_id="b",
                provider="test",
                model="test",
                criteria=[shared, _criterion(3)],
            ),
        ],
    )

    pooled = pool_criteria(ensemble)

    assert [criterion.title for _, criterion in pooled] == [
        "Title 1",
        "Title 2",
        "Title 3",
    ]
    assert [agent_id for agent_id, _ in pooled] == ["a", "a", "b"]


def test_dedupe_ignores_case_and_punctuation_differences() -> None:
    loud = GeneratedCriterion(
        title="Shouty",
        given="The User Is Logged In!",
        when="They Click Save.",
        then="The Record Persists?",
    )
    quiet = GeneratedCriterion(
        title="Quiet",
        given="the user is logged in",
        when="they click save",
        then="the record persists",
    )

    assert loud.dedupe_key() == quiet.dedupe_key()


# ---------------------------------------------------------------------------
# R11 - top-K retrieved context is injected into every agent's prompt
# ---------------------------------------------------------------------------


async def test_retrieval_asks_for_exactly_top_k_chunks() -> None:
    ingest = _FakeIngest(["chunk one", "chunk two", "chunk three"])
    service = GenerationService(_settings(rag_top_k=2), ingest_service=ingest)

    chunks = await service.retrieve_context("reset my password")

    assert ingest.calls == [("reset my password", 2)]
    assert chunks == ["chunk one", "chunk two"]


async def test_retrieval_is_skipped_when_top_k_is_zero() -> None:
    ingest = _FakeIngest(["chunk one"])
    service = GenerationService(_settings(rag_top_k=0), ingest_service=ingest)

    assert await service.retrieve_context("story") == []
    assert ingest.calls == []


async def test_retrieval_without_a_vector_store_returns_no_context() -> None:
    service = GenerationService(_settings(), ingest_service=None)

    assert await service.retrieve_context("story") == []


async def test_a_broken_vector_store_does_not_fail_generation() -> None:
    """Retrieval is best-effort: no context is better than no criteria."""

    class _BrokenIngest:
        def search(self, query: str, k: int = 5):
            raise ConnectionError("pgvector is down")

    service = GenerationService(_settings(), ingest_service=_BrokenIngest())

    assert await service.retrieve_context("story") == []


def test_retrieved_chunks_are_rendered_into_the_prompt() -> None:
    deps = GenerationDeps(
        user_story="story",
        context_chunks=["Passwords expire after 90 days.", "Reset links last 1 hour."],
    )

    block = deps.context_block()

    assert "[Context 1]" in block
    assert "[Context 2]" in block
    assert "Passwords expire after 90 days." in block
    assert "Reset links last 1 hour." in block


def test_absent_context_is_stated_rather_than_left_blank() -> None:
    """Silence would let the model assume context it never received."""
    block = GenerationDeps(user_story="story").context_block()

    assert "No project context documents were retrieved" in block


def test_reviewer_feedback_is_only_added_on_a_regeneration_pass() -> None:
    assert GenerationDeps(user_story="story").feedback_block() is None

    block = GenerationDeps(
        user_story="story", feedback="Cover the lockout case."
    ).feedback_block()

    assert block is not None
    assert "Cover the lockout case." in block


def test_user_prompt_carries_the_story_and_the_criteria_budget() -> None:
    prompt = build_user_prompt(GenerationDeps(user_story="As a user I want X"), 6)

    assert "up to 6 acceptance criteria" in prompt
    assert "As a user I want X" in prompt


# ---------------------------------------------------------------------------
# Conversation -> story composition
# ---------------------------------------------------------------------------


def test_only_user_turns_reach_the_agents() -> None:
    """Assistant replies are the system talking to itself; they would bias it."""
    story = compose_user_story(
        [
            ConversationMessage(
                role="user", text="As a user I want to reset my password"
            ),
            ConversationMessage(role="assistant", text="Sure, that sounds clear."),
            ConversationMessage(role="user", text="It must expire after one hour."),
        ]
    )

    assert "reset my password" in story
    assert "It must expire after one hour." in story
    assert "Sure, that sounds clear." not in story


def test_attachment_text_is_included_in_the_story() -> None:
    story = compose_user_story(
        [
            ConversationMessage(
                role="user",
                text="Here is the spec",
                attachments=[
                    ConversationAttachment(
                        filename="spec.pdf",
                        content="Reset links expire after 60 minutes.",
                    )
                ],
            )
        ]
    )

    assert "spec.pdf" in story
    assert "Reset links expire after 60 minutes." in story


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [ConversationMessage(role="user", text="   ")],
        [ConversationMessage(role="assistant", text="Only an assistant reply")],
    ],
)
def test_an_empty_conversation_is_refused_rather_than_invented(messages) -> None:
    """Asked to work from nothing, a model invents a feature. Fail instead."""
    with pytest.raises(NoUserStoryError):
        compose_user_story(messages)


# ---------------------------------------------------------------------------
# Scoring seam - delegated to app.services.scoring, which is shared with
# AC regeneration and UAT generation. That module owns the voting-layer
# contract (text matching, 0-5 clamping); these cover only this layer's use
# of it.
# ---------------------------------------------------------------------------


def test_every_criterion_is_scored_exactly_once() -> None:
    """A miscount here would silently shift scores onto the wrong criteria."""
    criteria = [_criterion(n) for n in range(1, 6)]
    scores = _fake_scorer([1.0, 2.0, 3.0, 4.0, 5.0])(
        prompt='p', candidates=['a', 'b', 'c', 'd', 'e']
    )

    ranked = _to_wire_models(criteria, scores=scores)

    assert len(ranked) == 5
    assert [item.overall_score for item in ranked] == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_scores_stay_within_the_apps_zero_to_five_scale() -> None:
    """The app model rejects anything above 5, so no rescaling may sneak in."""
    criteria = [_criterion(1)]
    scores = _fake_scorer([5.0])(prompt='p', candidates=['a'])

    ranked = _to_wire_models(criteria, scores=scores)

    assert ranked[0].overall_score == 5.0
    assert ranked[0].scores.correctness == 5.0


# --- Title-anchored generation -------------------------------------------
#
# The review finding these cover: pooling independent per-agent criteria left
# nothing to group equivalent candidates by, so near-duplicates survived and
# the voting layer compared unrelated items. Anchoring on one agent's titles
# gives every candidate a join key.


def _agent_result(agent_id: str, titles: list[str]) -> AgentResult:
    return AgentResult(
        agent_id=agent_id,
        provider="test",
        model="test",
        criteria=[
            GeneratedCriterion(
                title=title,
                given=f"{agent_id} precondition",
                when=f"{agent_id} action",
                then=f"{agent_id} outcome",
            )
            for title in titles
        ],
    )


def test_group_by_title_collects_one_candidate_per_agent():
    anchor = _agent_result("anchor", ["Log in", "Reject bad password"])
    follower = _agent_result("follower", ["Log in", "Reject bad password"])

    groups = orchestrator.group_by_title(anchor, [follower])

    assert [group.title for group in groups] == ["Log in", "Reject bad password"]
    assert all(len(group.candidates) == 2 for group in groups)
    assert [c.agent_id for c in groups[0].candidates] == ["anchor", "follower"]


def test_group_by_title_matches_despite_case_and_punctuation_drift():
    """Models reword titles slightly; the join must survive that."""
    anchor = _agent_result("anchor", ["Rate a finished book"])
    follower = _agent_result("follower", ["  rate a finished BOOK.  "])

    groups = orchestrator.group_by_title(anchor, [follower])

    assert len(groups) == 1
    assert len(groups[0].candidates) == 2
    # The anchor's spelling is the one shown, not the follower's.
    assert groups[0].title == "Rate a finished book"


def test_group_by_title_drops_follower_titles_that_match_nothing():
    """An invented title would reintroduce the ungrouped duplicates."""
    anchor = _agent_result("anchor", ["Log in"])
    follower = _agent_result("follower", ["Log in", "Something else entirely"])

    groups = orchestrator.group_by_title(anchor, [follower])

    assert len(groups) == 1
    assert len(groups[0].candidates) == 2


def test_group_by_title_ignores_a_followers_duplicate_answer():
    anchor = _agent_result("anchor", ["Log in"])
    follower = _agent_result("follower", ["Log in", "Log in"])

    groups = orchestrator.group_by_title(anchor, [follower])

    assert len(groups[0].candidates) == 2


def test_candidate_group_ranks_best_first():
    group = CandidateGroup(
        title="Log in",
        candidates=[
            Candidate(agent_id="a", criterion=_criterion(1), overall_score=2.0),
            Candidate(agent_id="b", criterion=_criterion(2), overall_score=4.5),
        ],
    )

    group.rank()

    assert group.winner.agent_id == "b"
    assert [c.agent_id for c in group.alternatives] == ["a"]


def test_candidate_group_tie_break_is_deterministic():
    """Equal scores must not resolve by whichever agent finished first."""
    def build(order: list[str]) -> CandidateGroup:
        group = CandidateGroup(
            title="Log in",
            candidates=[
                Candidate(agent_id=agent, criterion=_criterion(1), overall_score=3.0)
                for agent in order
            ],
        )
        group.rank()
        return group

    assert build(["zeta", "alpha"]).winner.agent_id == "alpha"
    assert build(["alpha", "zeta"]).winner.agent_id == "alpha"


def test_grouped_wire_models_expose_winner_with_alternatives_behind_it():
    group = CandidateGroup(
        title="Log in",
        candidates=[
            Candidate(agent_id="weak", criterion=_criterion(1), overall_score=1.0),
            Candidate(agent_id="strong", criterion=_criterion(2), overall_score=4.0),
        ],
    )
    group.rank()

    wire = _groups_to_wire_models([group])

    assert len(wire) == 1
    assert wire[0].title == "Log in"
    assert wire[0].source_agent == "strong"
    assert wire[0].overall_score == 4.0
    assert len(wire[0].alternatives) == 1
    assert wire[0].alternatives[0].source_agent == "weak"
    assert wire[0].alternatives[0].overall_score == 1.0


def test_grouped_wire_models_stay_backwards_compatible():
    """A client ignoring the new fields must see exactly one row per behaviour."""
    groups = [
        CandidateGroup(
            title=f"Title {n}",
            candidates=[
                Candidate(agent_id="a", criterion=_criterion(n), overall_score=3.0),
                Candidate(agent_id="b", criterion=_criterion(n), overall_score=1.0),
            ],
        )
        for n in (1, 2)
    ]
    for group in groups:
        group.rank()

    wire = _groups_to_wire_models(groups)

    assert len(wire) == 2
    assert [item.id for item in wire] == [1, 2]
    dumped = wire[0].model_dump()
    assert dumped["title"] and dumped["given"] and dumped["scores"]


def test_group_by_title_with_no_followers_still_produces_groups():
    """A single-agent roster must not be a special case."""
    anchor = _agent_result("anchor", ["Log in", "Log out"])

    groups = orchestrator.group_by_title(anchor, [])

    assert len(groups) == 2
    assert all(len(group.candidates) == 1 for group in groups)
    assert groups[0].alternatives == []
