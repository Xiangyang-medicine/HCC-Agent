"""Phase 4 closed-loop agent system.

This package contains the only canonical implementation for Phase 4.  It is
deliberately isolated from legacy rule-based risk evaluators.
"""

from .orchestrator import ClosedLoopAgent, SystemVariant
from .schema import TaskRequest, RunStatus
from .llm_policy import StructuredLLMPlanningPolicy, StructuredLLMSynthesisPolicy

__all__ = [
    "ClosedLoopAgent",
    "RunStatus",
    "StructuredLLMSynthesisPolicy",
    "StructuredLLMPlanningPolicy",
    "SystemVariant",
    "TaskRequest",
]
