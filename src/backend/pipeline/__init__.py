"""
pipeline — glue between the Ensemble Generation Layer and the Voting Layer.

Public API:
    build_evaluation_requests  EnsembleResult -> per-agent Voting Layer payloads
    map_votes_to_criteria      VotingResult   -> votes attributed to criterion ids
"""

from .adapter import (
    AgentEvaluationRequest,
    CriterionVote,
    build_evaluation_requests,
    map_votes_to_criteria,
)

__all__ = [
    "AgentEvaluationRequest",
    "CriterionVote",
    "build_evaluation_requests",
    "map_votes_to_criteria",
]
