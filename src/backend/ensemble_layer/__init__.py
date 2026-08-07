"""
ensemble — Ensemble Generation Layer package.

Public API (import from here, not from sub-modules):
    EnsembleOrchestrator  — runs N agents in parallel
    RunContext            — dependency injection container
    EnsembleConfig        — runtime configuration
    UserStory             — input model
    EnsembleResult        — output bundle (passed to Voting Layer)
"""

from .context import RunContext, VectorStoreStub
from .models import (
    AcceptanceCriteria,
    AgentConfig,
    AgentResult,
    EnsembleConfig,
    EnsembleResult,
    GenerationStatus,
    Provider,
    UserStory,
)
from .orchestrator import EnsembleOrchestrator

__all__ = [
    "EnsembleOrchestrator",
    "RunContext",
    "VectorStoreStub",
    "EnsembleConfig",
    "UserStory",
    "EnsembleResult",
    "AgentResult",
    "AcceptanceCriteria",
    "AgentConfig",
    "GenerationStatus",
    "Provider",
]