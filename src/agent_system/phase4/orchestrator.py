"""Closed-loop Phase 4 orchestration with observable revision and abstention."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from .schema import ActionPlan, Claim, DraftReport, RunState, RunStatus, TaskRequest, ToolStatus
from .tools import EvidenceTool, PrognosticTool
from .verifier import ReportVerifier


class SystemVariant(str, Enum):
    B0_ENGINE_ONLY = "B0_ENGINE_ONLY"
    B1_SINGLE_LLM_NO_TOOLS = "B1_SINGLE_LLM_NO_TOOLS"
    B2_SINGLE_LLM_WITH_TOOLS = "B2_SINGLE_LLM_WITH_TOOLS"
    B3_MULTI_AGENT_NO_VERIFIER = "B3_MULTI_AGENT_NO_VERIFIER"
    B4_FULL_CLOSED_LOOP = "B4_FULL_CLOSED_LOOP"


class SynthesisPolicy(Protocol):
    """Pluggable LLM policy boundary.

    Implementations receive only structured tool outputs.  They must return a
    draft; they may not create or alter numerical model output.
    """

    def create_draft(self, state: RunState, revision: bool) -> DraftReport: ...


class PlanningPolicy(Protocol):
    def create_plan(self, request: TaskRequest) -> ActionPlan: ...


class DeterministicPlanningPolicy:
    """Offline-only planner used for architecture tests, never formal results."""

    def create_plan(self, request: TaskRequest) -> ActionPlan:
        return ActionPlan(
            steps=("prognostic_tool", "evidence_tool"),
            arguments={
                "case_id": request.case_id,
                "repeat": request.repeat,
                "requested_model": request.requested_model,
            },
            status="OFFLINE_TEST_ONLY",
        )


class TemplateSynthesisPolicy:
    """Deterministic offline policy used only for architecture smoke tests."""

    def __init__(self, omit_citation_on_first_attempt: bool = False):
        self.omit_citation_on_first_attempt = omit_citation_on_first_attempt

    def create_draft(self, state: RunState, revision: bool) -> DraftReport:
        if state.model_result is None:
            raise ValueError("A model result is required before synthesis.")
        evidence = state.evidence_result
        citations: tuple[str, ...] = ()
        claim_text = "No biomedical context was retrieved."
        if evidence is not None and evidence.passages:
            passage = evidence.passages[0]
            claim_text = passage.text
            if revision or not self.omit_citation_on_first_attempt:
                citations = (passage.passage_id,)
        probabilities = state.model_result.survival_probabilities
        report = (
            "Technical research report only. "
            f"Frozen model={state.model_result.model_id}; "
            f"risk_score={state.model_result.risk_score:.12g}; "
            f"36-month survival probability={probabilities.get('36m', float('nan')):.12g}. "
            "No treatment advice is included."
        )
        return DraftReport(
            model_result=state.model_result,
            claims=(Claim(text=claim_text, citation_ids=citations),),
            report_text=report,
        )


class ClosedLoopAgent:
    """Canonical state machine: validate -> tools -> verify -> revise/report/abstain."""

    def __init__(
        self,
        prognostic_tool: PrognosticTool,
        evidence_tool: EvidenceTool,
        synthesis_policy: SynthesisPolicy,
        planning_policy: PlanningPolicy | None = None,
        verifier: ReportVerifier | None = None,
        variant: SystemVariant = SystemVariant.B4_FULL_CLOSED_LOOP,
        max_revisions: int = 1,
        enable_plan_revision: bool = True,
        enable_evidence_retry: bool = True,
        enable_synthesis_revision: bool = True,
    ):
        if max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        self.prognostic_tool = prognostic_tool
        self.evidence_tool = evidence_tool
        self.synthesis_policy = synthesis_policy
        self.planning_policy = planning_policy or DeterministicPlanningPolicy()
        self.verifier = verifier or ReportVerifier()
        self.variant = variant
        self.max_revisions = max_revisions
        self.enable_plan_revision = enable_plan_revision
        self.enable_evidence_retry = enable_evidence_retry
        self.enable_synthesis_revision = enable_synthesis_revision

    @staticmethod
    def _valid_request(request: TaskRequest) -> tuple[bool, str | None]:
        if not request.task_id or not request.case_id:
            return False, "task_id and case_id are required"
        if request.repeat < 1 or request.repeat > 5:
            return False, "repeat must be in the frozen Phase 3A range 1..5"
        if request.requested_model != "M4_combined_rsf":
            return False, "only the frozen provisional M4 evaluation tool is available"
        return True, None

    def run(self, request: TaskRequest) -> RunState:
        state = RunState(request=request)
        valid, reason = self._valid_request(request)
        if not valid:
            state.trace("validate", "INVALID_INPUT", reason=reason)
            state.status = RunStatus.SAFE_ABSTAIN
            state.abstention_reason = reason
            return state
        state.trace("validate", "SUCCESS", fields=sorted(request.clinical_fields))

        if self.variant == SystemVariant.B1_SINGLE_LLM_NO_TOOLS:
            state.trace("plan", "UNSUPPORTED", reason="B1 is not available in offline canonical mode")
            state.status = RunStatus.SAFE_ABSTAIN
            state.abstention_reason = "B1 requires an explicit external LLM implementation."
            return state

        state.plan = self.planning_policy.create_plan(request)
        required_steps = ("prognostic_tool", "evidence_tool")
        required_arguments = {
            "case_id": request.case_id,
            "repeat": request.repeat,
            "requested_model": request.requested_model,
        }
        plan_valid = (
            tuple(state.plan.steps) == required_steps
            and all(state.plan.arguments.get(key) == value for key, value in required_arguments.items())
        )
        state.trace(
            "plan",
            "SUCCESS" if plan_valid else "INVALID",
            plan=list(state.plan.steps),
            arguments=state.plan.arguments,
            planning_mode=(
                "single_tool_using_agent"
                if self.variant == SystemVariant.B2_SINGLE_LLM_WITH_TOOLS
                else "specialized_workflow"
            ),
        )
        if (
            not plan_valid
            and self.variant == SystemVariant.B4_FULL_CLOSED_LOOP
            and self.enable_plan_revision
            and hasattr(self.planning_policy, "revise_plan")
        ):
            state.trace(
                "verify_plan",
                "FAIL",
                finding_code="INVALID_TOOL_PLAN_OR_ARGUMENTS",
            )
            state.planning_revision_count += 1
            state.trace("replan", "RETRY", revision=state.planning_revision_count)
            state.plan = self.planning_policy.revise_plan(
                request, "INVALID_TOOL_PLAN_OR_ARGUMENTS"
            )
            plan_valid = (
                tuple(state.plan.steps) == required_steps
                and all(
                    state.plan.arguments.get(key) == value
                    for key, value in required_arguments.items()
                )
            )
            state.trace(
                "verify_plan",
                "PASS" if plan_valid else "FAIL",
                plan=list(state.plan.steps),
                arguments=state.plan.arguments,
            )
        if not plan_valid:
            state.status = RunStatus.SAFE_ABSTAIN
            state.abstention_reason = "Planner did not produce the required tool sequence and arguments."
            state.trace("safe_abstain", "SUCCESS", reason=state.abstention_reason)
            return state
        state.model_result = self.prognostic_tool.predict(request)
        state.trace(
            "prognostic_tool",
            state.model_result.status.value,
            model_id=state.model_result.model_id,
            provenance=state.model_result.provenance,
        )
        if state.model_result.status != ToolStatus.SUCCESS:
            state.status = RunStatus.SAFE_ABSTAIN
            state.abstention_reason = state.model_result.message or "Prognostic tool unavailable."
            state.trace("safe_abstain", "SUCCESS", reason=state.abstention_reason)
            return state

        if self.variant == SystemVariant.B0_ENGINE_ONLY:
            state.final_report = DraftReport(
                model_result=state.model_result,
                claims=(),
                report_text="Frozen prognostic tool output only; no LLM synthesis.",
                status="ENGINE_ONLY",
            )
            state.status = RunStatus.VERIFIED_REPORT
            state.trace("report", "SUCCESS", mode="engine_only")
            return state

        state.evidence_result = self.evidence_tool.retrieve(request)
        state.trace(
            "evidence_tool",
            state.evidence_result.status.value,
            corpus_sha256=state.evidence_result.corpus_sha256,
        )
        if (
            state.evidence_result.status != ToolStatus.SUCCESS
            and self.variant == SystemVariant.B4_FULL_CLOSED_LOOP
            and self.enable_evidence_retry
        ):
            state.trace(
                "detect_tool_failure",
                "SUCCESS",
                tool="evidence_tool",
                original_status=state.evidence_result.status.value,
            )
            state.trace("retry_evidence_tool", "RETRY", retry=1)
            state.evidence_result = self.evidence_tool.retrieve(request)
            state.trace(
                "evidence_tool",
                state.evidence_result.status.value,
                retry=1,
                corpus_sha256=state.evidence_result.corpus_sha256,
            )
        if state.evidence_result.status != ToolStatus.SUCCESS:
            state.status = RunStatus.SAFE_ABSTAIN
            state.abstention_reason = state.evidence_result.message or "Evidence tool unavailable."
            state.trace("safe_abstain", "SUCCESS", reason=state.abstention_reason)
            return state

        while True:
            state.draft = self.synthesis_policy.create_draft(
                state, revision=state.revision_count > 0
            )
            state.trace("synthesize", "SUCCESS", revision=state.revision_count)

            if self.variant in {
                SystemVariant.B2_SINGLE_LLM_WITH_TOOLS,
                SystemVariant.B3_MULTI_AGENT_NO_VERIFIER,
            }:
                state.final_report = state.draft
                state.status = RunStatus.VERIFIED_REPORT
                state.trace(
                    "report",
                    "UNVERIFIED",
                    reason=(
                        "B2 has no separate verifier/revision controller"
                        if self.variant == SystemVariant.B2_SINGLE_LLM_WITH_TOOLS
                        else "B3 verifier/revision ablation"
                    ),
                )
                return state

            state.verification = self.verifier.verify(
                state.draft, state.model_result, state.evidence_result
            )
            state.trace(
                "verify",
                "PASS" if state.verification.passed else "FAIL",
                finding_codes=[finding.code for finding in state.verification.findings],
            )
            if state.verification.passed:
                state.final_report = state.draft
                state.status = RunStatus.VERIFIED_REPORT
                state.trace("report", "SUCCESS", revision=state.revision_count)
                return state
            if not self.enable_synthesis_revision or state.revision_count >= self.max_revisions:
                state.status = RunStatus.SAFE_ABSTAIN
                state.abstention_reason = "Verifier rejected output after retry budget."
                state.trace("safe_abstain", "SUCCESS", reason=state.abstention_reason)
                return state
            state.revision_count += 1
            state.trace("revise", "RETRY", revision=state.revision_count)
