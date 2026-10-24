"""Offline tests for the ensemble generation layer, grouped by requirement."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import NativeOutput, PromptedOutput

from app.config import Settings
from app.generation import orchestrator
from app.generation.agents import build_agent
from app.generation.models import (
    AgentResult,
    Candidate,
    CandidateGroup,
    EnsembleResult,
    GeneratedCriteriaSet,
    GeneratedCriterion,
    NumberedCriteriaSet,
    NumberedCriterion,
)
from app.generation.orchestrator import (
    pool_criteria,
    run_ensemble,
    run_title_anchored_ensemble,
)
from app.generation.prompts import (
    GenerationDeps,
    build_descriptions_prompt,
    build_user_prompt,
)
from app.llm_config import (
    GenerationAgentConfig,
    ModelsConfigError,
    load_models_config,
)
from app.models_conversation import ConversationAttachment, ConversationMessage
from app.services.generation import (
    GenerationService,
    NoUserStoryError,
    _groups_to_wire_models,
    _to_wire_models,
    compose_user_story,
)
from app.services.scoring import CandidateScore
from llm.spec import ModelBuildError, with_output_mode


def _test_agent(agent_id: str) -> GenerationAgentConfig:
    """Return a TestModel roster entry, so no provider is contacted."""
    return GenerationAgentConfig(
        id=agent_id, provider="test", model="test", temperature=0.0, output_mode="tool"
    )


def _criterion(number: int) -> GeneratedCriterion:
    """Return a distinguishable criterion; assertions key on `title`."""
    return GeneratedCriterion(
        title=f"Title {number}",
        given=f"precondition {number}",
        when=f"action {number}",
        then=f"outcome {number}",
    )


def _settings(**overrides) -> Settings:
    """Return settings with explicit values, immune to a stray .env."""
    base: dict[str, Any] = dict(
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


class _FakeRetriever:
    """Records its calls so the top-K contract can be asserted."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[_FakeChunk]:
        self.calls.append((query, top_k))
        return [_FakeChunk(text) for text in self._contents[:top_k]]


def _fake_scorer(overalls: list[float]):
    """Return a scorer stub giving one `CandidateScore` per candidate, in order."""

    def scorer(prompt, candidates, *, reference_answer="", settings=None):
        assert len(candidates) == len(overalls), "scorer got the wrong candidate count"
        return [
            CandidateScore(
                relevance=overall,
                correctness=overall,
                understandability=overall,
                coverage=overall,
                overall=overall,
            )
            for overall in overalls
        ]

    return scorer


@pytest.mark.asyncio
async def test_agents_run_concurrently_not_serially(monkeypatch) -> None:
    """Total time should match the slowest agent, not the sum."""

    async def slow_run(config, deps, *, max_criteria, timeout_seconds):
        await asyncio.sleep(0.3)
        return orchestrator.AgentResult(
            agent_id=config.id, provider=config.provider, model=config.model
        )

    monkeypatch.setattr(orchestrator, "_run_one", slow_run)

    agents = [_test_agent(f"a{index}") for index in range(5)]
    started = time.perf_counter()
    await run_ensemble(GenerationDeps(user_story="story"), agents=agents)
    elapsed = time.perf_counter() - started

    # Serial would be ~1.5s; concurrent should be ~0.3s.
    assert elapsed < 0.9, f"agents appear to be running serially ({elapsed:.2f}s)"


def test_scores_land_on_their_own_criterion_and_rank_highest_first() -> None:
    """Each criterion keeps its own score, and the strongest sorts first."""
    criteria = [_criterion(1), _criterion(2), _criterion(3)]
    scores = _fake_scorer([2.0, 5.0, 1.0])(prompt="p", candidates=["a", "b", "c"])

    ranked = _to_wire_models(criteria, scores=scores)

    assert [item.title for item in ranked] == ["Title 2", "Title 1", "Title 3"]
    assert [item.overall_score for item in ranked] == [5.0, 2.0, 1.0]
    assert ranked[0].scores.relevance == 5.0


def test_unscored_criteria_are_returned_when_voting_is_disabled() -> None:
    """Voting off drops the scores, never the criteria."""
    criteria = [_criterion(1), _criterion(2)]

    wired = _to_wire_models(criteria, scores=None)

    assert len(wired) == 2
    assert all(item.overall_score == 0.0 for item in wired)
    assert all(item.scores.coverage == 0.0 for item in wired)


# R7: generated criteria must be valid Gherkin.


def test_criterion_requires_all_three_gherkin_clauses() -> None:
    with pytest.raises(ValidationError):
        GeneratedCriterion.model_validate(
            {
                "title": "Missing the then clause",
                "given": "a registered user is on the login page",
                "when": "they submit valid credentials",
            }
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
    """An empty set counts as a failure."""
    with pytest.raises(ValidationError):
        GeneratedCriteriaSet(user_story_summary="A summary.", criteria=[])


def test_criteria_set_caps_the_number_of_criteria() -> None:
    with pytest.raises(ValidationError):
        GeneratedCriteriaSet(
            user_story_summary="A summary.",
            criteria=[_criterion(number) for number in range(11)],
        )


async def test_generated_criteria_survive_a_full_ensemble_run() -> None:
    """On TestModel, every pooled criterion is valid Gherkin."""
    ensemble = await run_ensemble(
        GenerationDeps(user_story="As a user I want to log in"),
        agents=[_test_agent("a"), _test_agent("b")],
    )

    pooled = pool_criteria(ensemble)

    assert pooled, "expected at least one candidate from the test roster"
    for _, criterion in pooled:
        assert criterion.given and criterion.when and criterion.then


# R8: the agent pool is declared in a config file.


QWEN = {"id": "qwen", "provider": "ollama", "model": "qwen2.5:7b"}
JUDGE = {"id": "claude", "provider": "anthropic", "model": "claude-sonnet-5-5"}


def _write_config(
    tmp_path, generation_agents: list[dict], judges: list[dict] | None = None, **extra
):
    """Write a models.json with the given agents and judges; return its path."""
    payload = {
        "chat_model": {"provider": "anthropic", "model": "claude-sonnet-5-5"},
        "vision_model": {"provider": "anthropic", "model": "claude-haiku-4-5"},
        "generation_agents": generation_agents,
        "judges": [JUDGE] if judges is None else judges,
        **extra,
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_models_config_loads_every_section_from_one_file(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        [QWEN, {"id": "llama", "provider": "ollama", "model": "llama3.1:8b"}],
    )

    config = load_models_config(path)

    assert config.chat_model.name == "anthropic:claude-sonnet-5-5"
    assert config.vision_model.name == "anthropic:claude-haiku-4-5"
    assert [agent.id for agent in config.generation_agents] == ["qwen", "llama"]
    assert config.generation_agents[0].model == "qwen2.5:7b"
    assert [judge.id for judge in config.judges] == ["claude"]


def test_models_config_returns_only_enabled_agents_and_judges(tmp_path) -> None:
    """Adding or retiring a model is a config edit, not code."""
    path = _write_config(
        tmp_path,
        [
            QWEN,
            {
                "id": "gemini",
                "provider": "google",
                "model": "gemini-2.0-flash",
                "enabled": False,
            },
        ],
        judges=[
            JUDGE,
            {
                "id": "luna",
                "provider": "openai",
                "model": "gpt-6-luna",
                "enabled": False,
            },
        ],
    )

    config = load_models_config(path)

    assert [agent.id for agent in config.enabled_agents()] == ["qwen"]
    assert [judge.id for judge in config.enabled_judges()] == ["claude"]


def test_temperature_is_omitted_unless_set(tmp_path) -> None:
    """Newer Claude and OpenAI reasoning models reject non-defaults."""
    config = load_models_config(_write_config(tmp_path, [QWEN]))

    assert config.generation_agents[0].temperature is None
    assert config.judges[0].temperature is None


def test_output_mode_defaults_to_native_and_is_read_per_entry(tmp_path) -> None:
    config = load_models_config(
        _write_config(
            tmp_path,
            [QWEN, {**QWEN, "id": "small", "output_mode": "prompted"}],
            judges=[{**JUDGE, "output_mode": "tool"}],
        )
    )

    assert [agent.output_mode for agent in config.generation_agents] == [
        "native",
        "prompted",
    ]
    assert config.chat_model.output_mode == "native"
    assert config.judges[0].output_mode == "tool"


def test_output_mode_wraps_the_output_type() -> None:
    """Sonnet 5.5 rejects a forced tool call, so needs native output."""
    assert with_output_mode(GeneratedCriteriaSet, "tool") is GeneratedCriteriaSet
    assert isinstance(with_output_mode(GeneratedCriteriaSet, "native"), NativeOutput)
    assert isinstance(
        with_output_mode(GeneratedCriteriaSet, "prompted"), PromptedOutput
    )
    assert with_output_mode(str, "native") is str


def test_models_config_missing_file_names_the_path_and_the_fix(tmp_path) -> None:
    with pytest.raises(ModelsConfigError) as error:
        load_models_config(tmp_path / "absent.json")

    assert "models.example.json" in str(error.value)


def test_models_config_rejects_malformed_json(tmp_path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ModelsConfigError, match="not valid JSON"):
        load_models_config(path)


def test_models_config_rejects_unknown_keys(tmp_path) -> None:
    """The old roster's `agents` key is reported, not ignored."""
    path = _write_config(tmp_path, [QWEN], agents=[QWEN])

    with pytest.raises(ModelsConfigError, match="invalid"):
        load_models_config(path)


def test_models_config_rejects_duplicate_agent_ids(tmp_path) -> None:
    """IDs say which agent wrote each candidate, so they must be unique."""
    path = _write_config(tmp_path, [QWEN, {**QWEN, "model": "qwen2.5:14b"}])

    with pytest.raises(ModelsConfigError, match="Duplicate agent ids"):
        load_models_config(path)


def test_models_config_rejects_duplicate_judge_ids(tmp_path) -> None:
    path = _write_config(tmp_path, [QWEN], judges=[JUDGE, JUDGE])

    with pytest.raises(ModelsConfigError, match="Duplicate judge ids"):
        load_models_config(path)


def test_models_config_rejects_nothing_enabled(tmp_path) -> None:
    no_agents = _write_config(tmp_path, [{**QWEN, "enabled": False}])
    with pytest.raises(ModelsConfigError, match="No enabled generation_agents"):
        load_models_config(no_agents)

    no_judges = _write_config(tmp_path, [QWEN], judges=[{**JUDGE, "enabled": False}])
    with pytest.raises(ModelsConfigError, match="No enabled judges"):
        load_models_config(no_judges)


def test_api_keys_are_read_from_the_environment_not_the_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file names the variable; its value is read at call time."""
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

    with pytest.raises(ModelBuildError, match="A_TEST_KEY"):
        build_agent(config)


# R10: agents run concurrently, and one bad agent doesn't fail the request.


class _ExplodingAgent:
    """An agent whose every run raises."""

    async def run(self, prompt, deps=None):
        raise RuntimeError("upstream exploded")


class _HangingAgent:
    """An agent that runs far past any test timeout."""

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
    error = ensemble.failed[0].error
    assert error is not None
    assert "upstream exploded" in error
    assert pool_criteria(ensemble), "the healthy agent's work must survive"


async def test_a_hanging_agent_is_cut_off_at_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One slow model must not use up the R6 time limit."""
    monkeypatch.setattr(orchestrator, "build_agent", lambda config: _HangingAgent())

    ensemble = await run_ensemble(
        GenerationDeps(user_story="story"),
        agents=[_test_agent("slow")],
        timeout_seconds=0.05,
    )

    assert ensemble.successful == []
    error = ensemble.failed[0].error
    assert error is not None
    assert "Timed out" in error


async def test_every_agent_failing_is_reported_as_a_failed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service reports every agent failing as an error."""
    monkeypatch.setattr(orchestrator, "build_agent", lambda config: _ExplodingAgent())

    service = GenerationService(_settings())
    messages = [ConversationMessage(role="user", text="As a user I want to log in")]

    with pytest.raises(RuntimeError, match="Every generation agent failed"):
        await service.generate(messages)


def test_pooling_drops_criteria_two_agents_produced_identically() -> None:
    """Identical criteria from several agents appear once."""
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


# R11: top-K retrieved context reaches every agent's prompt.


async def test_retrieval_asks_for_exactly_top_k_chunks() -> None:
    retriever = _FakeRetriever(["chunk one", "chunk two", "chunk three"])
    service = GenerationService(_settings(rag_top_k=2), context_retriever=retriever)

    chunks = await service.retrieve_context("reset my password")

    assert retriever.calls == [("reset my password", 2)]
    assert chunks == ["chunk one", "chunk two"]


async def test_retrieval_is_skipped_when_top_k_is_zero() -> None:
    retriever = _FakeRetriever(["chunk one"])
    service = GenerationService(_settings(rag_top_k=0), context_retriever=retriever)

    assert await service.retrieve_context("story") == []
    assert retriever.calls == []


async def test_retrieval_without_a_vector_store_returns_no_context() -> None:
    service = GenerationService(_settings(), context_retriever=None)

    assert await service.retrieve_context("story") == []


async def test_a_broken_vector_store_does_not_fail_generation() -> None:
    """A retrieval failure gives no context instead of an error."""

    class _BrokenRetriever:
        def search(self, query: str, top_k: int = 5):
            raise ConnectionError("pgvector is down")

    service = GenerationService(_settings(), context_retriever=_BrokenRetriever())

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
    """Say so, or the model may assume context it never received."""
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


# Composing the conversation into a user story.


def test_only_user_turns_reach_the_agents() -> None:
    """Assistant turns are left out."""
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
    """Fail instead of letting the model invent a feature."""
    with pytest.raises(NoUserStoryError):
        compose_user_story(messages)


# How generation uses scores. test_scoring.py covers the voting layer.


def test_every_criterion_is_scored_exactly_once() -> None:
    """A miscount would shift scores onto the wrong criteria."""
    criteria = [_criterion(number) for number in range(1, 6)]
    scores = _fake_scorer([1.0, 2.0, 3.0, 4.0, 5.0])(
        prompt="p", candidates=["a", "b", "c", "d", "e"]
    )

    ranked = _to_wire_models(criteria, scores=scores)

    assert len(ranked) == 5
    assert [item.overall_score for item in ranked] == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_scores_stay_within_the_apps_zero_to_five_scale() -> None:
    """The app model rejects values above 5, so nothing may rescale."""
    criteria = [_criterion(1)]
    scores = _fake_scorer([5.0])(prompt="p", candidates=["a"])

    ranked = _to_wire_models(criteria, scores=scores)

    assert ranked[0].overall_score == 5.0
    assert ranked[0].scores.correctness == 5.0


# Title-anchored generation.


def _agent_result(agent_id: str, titles: list[str]) -> AgentResult:
    """Return a successful result with one criterion per title."""
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
    assert [candidate.agent_id for candidate in groups[0].candidates] == [
        "anchor",
        "follower",
    ]


def test_group_by_title_matches_despite_case_and_punctuation_drift():
    """Titles differing only in case, spacing or punctuation still match."""
    anchor = _agent_result("anchor", ["Rate a finished book"])
    follower = _agent_result("follower", ["  rate a finished BOOK.  "])

    groups = orchestrator.group_by_title(anchor, [follower])

    assert len(groups) == 1
    assert len(groups[0].candidates) == 2
    # The anchor's title is shown.
    assert groups[0].title == "Rate a finished book"


def test_group_by_title_drops_follower_titles_that_match_nothing():
    """Follower titles with no anchor match are dropped."""
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
    assert [candidate.agent_id for candidate in group.alternatives] == ["a"]


def test_candidate_group_tie_break_is_deterministic():
    """Equal scores break ties by agent ID, not finishing order."""

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
    """A client ignoring `alternatives` sees one row per behaviour."""
    groups = [
        CandidateGroup(
            title=f"Title {number}",
            candidates=[
                Candidate(
                    agent_id="a", criterion=_criterion(number), overall_score=3.0
                ),
                Candidate(
                    agent_id="b", criterion=_criterion(number), overall_score=1.0
                ),
            ],
        )
        for number in (1, 2)
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


def test_unrated_candidate_ranks_below_a_genuinely_bad_one():
    """An unscored 0.0 ranks below a real 0.0."""
    group = CandidateGroup(
        title="Log in",
        candidates=[
            Candidate(
                agent_id="aaa-unrated",
                criterion=_criterion(1),
                overall_score=0.0,
                scoring_failed=True,
            ),
            Candidate(
                agent_id="zzz-rated-badly",
                criterion=_criterion(2),
                overall_score=0.0,
                scoring_failed=False,
            ),
        ],
    )

    group.rank()

    # "aaa" would win the tie-break, but unrated candidates go last.
    assert group.winner.agent_id == "zzz-rated-badly"
    assert [candidate.agent_id for candidate in group.unrated] == ["aaa-unrated"]


def test_unrated_candidate_never_beats_a_rated_one():
    group = CandidateGroup(
        title="Log in",
        candidates=[
            Candidate(
                agent_id="a",
                criterion=_criterion(1),
                overall_score=5.0,
                scoring_failed=True,
            ),
            Candidate(
                agent_id="b",
                criterion=_criterion(2),
                overall_score=1.0,
                scoring_failed=False,
            ),
        ],
    )

    group.rank()

    assert group.winner.agent_id == "b"


# Pass 2 answers by title number, as reworded titles wouldn't match.


def _numbered(*numbers: int) -> NumberedCriteriaSet:
    """Return pass-2 answers for the given title numbers, in that order."""
    return NumberedCriteriaSet(
        criteria=[
            NumberedCriterion(
                title_number=number,
                given=f"given {number}",
                when=f"when {number}",
                then=f"then {number}",
            )
            for number in numbers
        ]
    )


def test_numbered_answers_carry_the_anchors_exact_titles():
    titles = ["Log in", "Reject bad password"]

    criteria = orchestrator._attach_anchor_titles("follower", _numbered(2, 1), titles)

    assert [(criterion.title, criterion.given) for criterion in criteria] == [
        ("Reject bad password", "given 2"),
        ("Log in", "given 1"),
    ]


def test_unknown_and_repeated_title_numbers_are_dropped():
    titles = ["Log in", "Reject bad password"]

    criteria = orchestrator._attach_anchor_titles(
        "follower", _numbered(1, 3, 1, 2), titles
    )

    assert [criterion.title for criterion in criteria] == [
        "Log in",
        "Reject bad password",
    ]
    assert criteria[0].given == "given 1"


def test_descriptions_prompt_asks_for_numbers_not_copied_titles():
    prompt = build_descriptions_prompt(
        GenerationDeps(user_story="story"), ["Log in", "Log out"]
    )

    assert "1. Log in" in prompt and "2. Log out" in prompt
    assert "title_number" in prompt
    assert "Copy each title" not in prompt


async def test_title_anchored_ensemble_groups_every_follower_answer():
    ensemble, groups = await run_title_anchored_ensemble(
        GenerationDeps(user_story="story"),
        agents=[_test_agent("anchor"), _test_agent("follower")],
    )

    assert all(result.ok for result in ensemble.results)
    # Matched by number, including the title unique to the anchor.
    assert [group.title for group in groups] == [
        "Successful login redirects home",
        "Invalid password shows an error",
        "Unique candidate from anchor",
    ]
    for group in groups:
        assert sorted(candidate.agent_id for candidate in group.candidates) == [
            "anchor",
            "follower",
        ]
    third = next(
        candidate
        for candidate in groups[2].candidates
        if candidate.agent_id == "follower"
    )
    assert third.criterion.given == "the follower agent produced this candidate"
