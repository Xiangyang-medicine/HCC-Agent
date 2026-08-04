"""External scoring and comparator execution for the Phase 4 benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from enum import Enum
from typing import Any

from .llm_policy import StructuredLLMPlanningPolicy, StructuredLLMSynthesisPolicy, parse_json_object
from .orchestrator import (
    ClosedLoopAgent,
    DeterministicPlanningPolicy,
    SystemVariant,
    TemplateSynthesisPolicy,
)
from .schema import ActionTrace, Claim, DraftReport, RunState, RunStatus, TaskRequest
from .tools import EvidenceTool, PrognosticTool
from .verifier import FORBIDDEN_PHRASES


AGENT_REQUIRED_TOOLS = ("prognostic_tool", "evidence_tool")


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def serializable_state(state: RunState) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(state), default=_primitive, ensure_ascii=False))


def run_b1_no_tools(request: TaskRequest, completion) -> RunState:
    """Run the no-tool LLM baseline without injecting model or evidence data."""
    state = RunState(request=request)
    state.trace("validate", "SUCCESS", mode="single_llm_no_tools")
    state.trace("plan", "NO_TOOLS_AVAILABLE", plan=[])
    system = """You are a single language model answering a research-only prognosis task.
No tools, model output, patient outcomes, or evidence passages are available.
Return JSON only with keys `claims` and `narrative`. Claims must be a list of
objects with `text` and `citation_ids`. Do not make treatment recommendations."""
    payload = {
        "task_id": request.task_id,
        "case_id": request.case_id,
        "request": request.query,
    }
    try:
        parsed = parse_json_object(completion(system, json.dumps(payload, ensure_ascii=False)))
        raw_claims = parsed.get("claims", [])
        if not isinstance(raw_claims, list):
            raw_claims = []
        claims = tuple(
            Claim(
                text=str(item["text"]),
                citation_ids=tuple(str(value) for value in item.get("citation_ids", [])),
            )
            for item in raw_claims
            if isinstance(item, dict) and "text" in item
        )
        narrative = str(parsed.get("narrative", ""))
        state.trace("synthesize", "SUCCESS")
    except Exception as exc:
        claims = ()
        narrative = f"Unstructured baseline failure: {type(exc).__name__}."
        state.trace("synthesize", "FAILED", error_type=type(exc).__name__)
    state.final_report = DraftReport(
        model_result=None,
        claims=claims,
        report_text=narrative,
        status="UNVERIFIED_NO_TOOLS",
    )
    state.status = RunStatus.VERIFIED_REPORT
    state.trace("report", "UNVERIFIED", reason="B1 has no tools or external verifier")
    return state


def run_comparator(
    request: TaskRequest,
    variant: SystemVariant,
    prognostic_tool: PrognosticTool,
    evidence_tool: EvidenceTool,
    completion,
) -> RunState:
    if variant == SystemVariant.B1_SINGLE_LLM_NO_TOOLS:
        return run_b1_no_tools(request, completion)
    if variant == SystemVariant.B0_ENGINE_ONLY:
        return ClosedLoopAgent(
            prognostic_tool=prognostic_tool,
            evidence_tool=evidence_tool,
            synthesis_policy=TemplateSynthesisPolicy(),
            planning_policy=DeterministicPlanningPolicy(),
            variant=variant,
        ).run(request)
    return ClosedLoopAgent(
        prognostic_tool=prognostic_tool,
        evidence_tool=evidence_tool,
        planning_policy=StructuredLLMPlanningPolicy(completion),
        synthesis_policy=StructuredLLMSynthesisPolicy(completion),
        variant=variant,
        max_revisions=1,
    ).run(request)


def score_clean_run(
    state: RunState,
    variant: SystemVariant,
    request: TaskRequest,
    prognostic_tool: PrognosticTool,
    evidence_tool: EvidenceTool,
) -> dict[str, Any]:
    expected_model = prognostic_tool.predict(request)
    expected_evidence = evidence_tool.retrieve(request)
    valid_passages = {p.passage_id: p.text for p in expected_evidence.passages}
    observed_tools = tuple(
        trace.step
        for trace in state.traces
        if trace.step in AGENT_REQUIRED_TOOLS and trace.status == "SUCCESS"
    )
    observed_unique = set(observed_tools)
    required = set(AGENT_REQUIRED_TOOLS)
    tool_precision = len(observed_unique & required) / len(observed_unique) if observed_unique else 0.0
    tool_recall = len(observed_unique & required) / len(required)
    tool_f1 = (
        2 * tool_precision * tool_recall / (tool_precision + tool_recall)
        if tool_precision + tool_recall else 0.0
    )
    exact_order = observed_tools == AGENT_REQUIRED_TOOLS

    plan_valid = False
    if state.plan is not None:
        plan_valid = (
            tuple(state.plan.steps) == AGENT_REQUIRED_TOOLS
            and state.plan.arguments.get("case_id") == request.case_id
            and state.plan.arguments.get("repeat") == request.repeat
            and state.plan.arguments.get("requested_model") == request.requested_model
        )

    report = state.final_report
    schema_valid = report is not None and isinstance(report.report_text, str)
    numeric_fidelity = False
    if report is not None and report.model_result == expected_model and expected_model.risk_score is not None:
        expected_risk = f"{expected_model.risk_score:.12g}"
        expected_36m = f"{expected_model.survival_probabilities.get('36m', float('nan')):.12g}"
        numeric_fidelity = (
            expected_model.model_id in report.report_text
            and expected_risk in report.report_text
            and expected_36m in report.report_text
        )

    claims = () if report is None else report.claims
    supported_flags = []
    citation_valid_flags = []
    citation_present_flags = []
    for claim in claims:
        citation_present_flags.append(bool(claim.citation_ids))
        citation_valid = bool(claim.citation_ids) and all(
            citation_id in valid_passages for citation_id in claim.citation_ids
        )
        citation_valid_flags.append(citation_valid)
        supported_flags.append(
            citation_valid and any(
                claim.text.strip() in valid_passages[citation_id]
                for citation_id in claim.citation_ids
            )
        )
    claim_count = len(claims)
    supported_precision = sum(supported_flags) / claim_count if claim_count else 0.0
    citation_completeness = sum(citation_present_flags) / claim_count if claim_count else 0.0
    citation_correctness = sum(citation_valid_flags) / claim_count if claim_count else 0.0
    unsupported_claim_rate = 1.0 - supported_precision if claim_count else 0.0

    report_lower = "" if report is None else report.report_text.lower()
    forbidden_claim = any(phrase in report_lower for phrase in FORBIDDEN_PHRASES)
    terminal_verified = state.status == RunStatus.VERIFIED_REPORT
    external_verifier_passed = bool(
        state.verification is not None and state.verification.passed
    )

    # B0 is a quantitative reference, not an agent. Its task success remains
    # false by design because it cannot satisfy evidence-grounded reporting.
    verified_task_success = bool(
        variant not in {SystemVariant.B0_ENGINE_ONLY, SystemVariant.B1_SINGLE_LLM_NO_TOOLS}
        and plan_valid
        and exact_order
        and schema_valid
        and numeric_fidelity
        and claim_count > 0
        and math.isclose(supported_precision, 1.0)
        and math.isclose(citation_completeness, 1.0)
        and math.isclose(citation_correctness, 1.0)
        and not forbidden_claim
        and terminal_verified
    )
    return {
        "verified_task_success": verified_task_success,
        "plan_valid": plan_valid,
        "tool_order_exact": exact_order,
        "tool_selection_precision": tool_precision,
        "tool_selection_recall": tool_recall,
        "tool_selection_f1": tool_f1,
        "schema_valid": schema_valid,
        "numeric_fidelity": numeric_fidelity,
        "claim_count": claim_count,
        "supported_claim_precision": supported_precision,
        "citation_completeness": citation_completeness,
        "citation_correctness": citation_correctness,
        "unsupported_claim_rate": unsupported_claim_rate,
        "forbidden_claim_case": forbidden_claim,
        "external_verifier_passed": external_verifier_passed,
        "safe_abstain": state.status == RunStatus.SAFE_ABSTAIN,
        "revision_count": state.revision_count,
    }
