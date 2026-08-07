"""
Adapter: Ensemble Generation Layer -> Voting Layer.

The two layers speak different shapes:

    EnsembleResult   one run  x N agents x M structured AcceptanceCriteria
    EvaluationInput  one (ai, model) x a flat list[str] of outputs

So one EnsembleResult fans out into N EvaluationInputs — one per agent whose
run produced usable criteria. Because EvaluationInput carries only strings,
the AcceptanceCriteria ids are lost in transit; the Voting Layer refers to a
criterion solely by its `output_index`. AgentEvaluationRequest keeps the
positional id map alongside the payload so votes can be attributed back to the
criterion that earned them.

This module maps and attributes only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ensemble_layer.models import (
    AgentConfig,
    AgentResult,
    EnsembleResult,
    GenerationStatus,
)
from voting.models import EvaluationInput, VotingResult


class AgentEvaluationRequest(BaseModel):
    """One agent's criteria, packaged for the Voting Layer.

    `criterion_ids` is positional: `criterion_ids[i]` is the
    AcceptanceCriteria.id behind `evaluation_input.output[i]`.
    """

    run_id: str
    agent_id: int
    evaluation_input: EvaluationInput
    criterion_ids: list[str] = Field(min_length=1)

    def criterion_id_for(self, output_index: int) -> str:
        """Resolve a Vote's `output_index` back to its criterion id."""
        try:
            return self.criterion_ids[output_index]
        except IndexError as exc:
            raise IndexError(
                f"output_index {output_index} is out of range for agent "
                f"#{self.agent_id}, which submitted {len(self.criterion_ids)} criteria"
            ) from exc


class CriterionVote(BaseModel):
    """A single rubric score, re-attached to the criterion it belongs to."""

    run_id: str
    agent_id: int
    criterion_id: str
    rubric: str
    score: int
    feedback: str
    is_error: bool = Field(
        False,
        description=(
            "True when the Voting Layer failed to score this criterion and "
            "emitted a sentinel instead of a real 1-5 judgement. Aggregation "
            "must exclude these rather than average them in."
        ),
    )


def _agents_by_id(config_agents: list[AgentConfig]) -> dict[int, AgentConfig]:
    by_id: dict[int, AgentConfig] = {}
    for agent_cfg in config_agents:
        if agent_cfg.agent_id in by_id:
            raise ValueError(
                f"Duplicate agent_id {agent_cfg.agent_id} in EnsembleConfig — "
                "agent ids must be unique to attribute votes."
            )
        by_id[agent_cfg.agent_id] = agent_cfg
    return by_id


def _is_evaluable(result: AgentResult) -> bool:
    """Whether an agent's output can be meaningfully scored.

    Three ways an AgentResult reaches the Voting Layer unusable:
      - the run failed or timed out;
      - it succeeded but returned no criteria (EvaluationInput requires >= 1);
      - it carries no prompt_used, which happens when the agent raised before
        its prompt was built. The rubrics substitute that prompt as
        {orig_instruction}, so an empty one yields meaningless scores.
    """
    return (
        result.status == GenerationStatus.SUCCESS
        and bool(result.criteria)
        and bool(result.prompt_used.strip())
    )


def build_evaluation_requests(result: EnsembleResult) -> list[AgentEvaluationRequest]:
    """Fan one ensemble run out into per-agent Voting Layer payloads.

    Agents whose output cannot be scored are skipped, so the returned list may
    be shorter than the ensemble — and empty if every agent failed. Callers
    that need a quorum should check the length.
    """
    agents_by_id = _agents_by_id(result.config.agents)
    requests: list[AgentEvaluationRequest] = []

    for agent_result in result.agent_results:
        if not _is_evaluable(agent_result):
            continue

        agent_cfg = agents_by_id.get(agent_result.agent_id)
        if agent_cfg is None:
            raise ValueError(
                f"AgentResult references agent_id {agent_result.agent_id}, "
                "which is absent from the run's EnsembleConfig."
            )

        requests.append(
            AgentEvaluationRequest(
                run_id=result.run_id,
                agent_id=agent_result.agent_id,
                evaluation_input=EvaluationInput(
                    ai=agent_cfg.provider.value,
                    model=agent_cfg.model,
                    prompt=agent_result.prompt_used,
                    output=[ac.to_gherkin() for ac in agent_result.criteria],
                ),
                criterion_ids=[ac.id for ac in agent_result.criteria],
            )
        )

    return requests


def map_votes_to_criteria(
    request: AgentEvaluationRequest,
    voting_result: VotingResult,
) -> list[CriterionVote]:
    """Re-attach a VotingResult's votes to the criteria that produced them.

    Scores outside the rubric's 1-5 range are the Voting Layer's error
    sentinels, not judgements; they are flagged rather than dropped so failures
    stay visible in the run log.
    """
    return [
        CriterionVote(
            run_id=request.run_id,
            agent_id=request.agent_id,
            criterion_id=request.criterion_id_for(vote.output_index),
            rubric=vote.rubric,
            score=vote.score,
            feedback=vote.feedback,
            is_error=not 1 <= vote.score <= 5,
        )
        for vote in voting_result.votes
    ]
