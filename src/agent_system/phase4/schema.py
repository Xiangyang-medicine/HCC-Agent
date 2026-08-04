"""Typed, observable state for the Phase 4 agent benchmark.

Only structured actions and data are recorded.  Private model reasoning is
not requested, stored, or evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    VERIFIED_REPORT = "VERIFIED_REPORT"
    SAFE_ABSTAIN = "SAFE_ABSTAIN"
    FAILED = "FAILED"


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True)
class TaskRequest:
    """A benchmark request.

    ``case_id`` and ``repeat`` identify one frozen OOF prediction.  The agent
    never receives survival outcome fields.
    """

    task_id: str
    case_id: str
    repeat: int
    requested_model: str = "M4_combined_rsf"
    clinical_fields: dict[str, Any] = field(default_factory=dict)
    query: str = "Summarize the model result with source-grounded technical context."


@dataclass(frozen=True)
class ActionTrace:
    step: str
    status: str
    timestamp_utc: str
    details: dict[str, Any]

    @classmethod
    def create(cls, step: str, status: str, **details: Any) -> "ActionTrace":
        return cls(
            step=step,
            status=status,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            details=details,
        )


@dataclass(frozen=True)
class ActionPlan:
    steps: tuple[str, ...]
    arguments: dict[str, Any]
    status: str = "PROPOSED"


@dataclass(frozen=True)
class ModelResult:
    status: ToolStatus
    case_id: str
    repeat: int
    model_id: str
    risk_score: float | None
    survival_probabilities: dict[str, float]
    source_sha256: str
    provenance: str
    message: str | None = None


@dataclass(frozen=True)
class EvidencePassage:
    source_id: str
    passage_id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class EvidenceResult:
    status: ToolStatus
    passages: tuple[EvidencePassage, ...]
    corpus_sha256: str
    message: str | None = None


@dataclass(frozen=True)
class Claim:
    text: str
    citation_ids: tuple[str, ...]
    kind: str = "biomedical_context"


@dataclass(frozen=True)
class DraftReport:
    model_result: ModelResult | None
    claims: tuple[Claim, ...]
    report_text: str
    status: str = "DRAFT"


@dataclass(frozen=True)
class VerificationFinding:
    code: str
    message: str
    blocking: bool = True


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    findings: tuple[VerificationFinding, ...]


@dataclass
class RunState:
    request: TaskRequest
    traces: list[ActionTrace] = field(default_factory=list)
    plan: ActionPlan | None = None
    model_result: ModelResult | None = None
    evidence_result: EvidenceResult | None = None
    draft: DraftReport | None = None
    verification: VerificationResult | None = None
    planning_revision_count: int = 0
    revision_count: int = 0
    status: RunStatus | None = None
    final_report: DraftReport | None = None
    abstention_reason: str | None = None

    def trace(self, step: str, status: str, **details: Any) -> None:
        self.traces.append(ActionTrace.create(step, status, **details))

    def serializable(self) -> dict[str, Any]:
        """Return only observable benchmark state, never private reasoning."""
        return asdict(self)
