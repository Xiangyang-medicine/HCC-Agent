"""Tests for the canonical Phase 4 architecture.

The tests are offline and deterministic.  They establish system behavior, not
formal LLM-benchmark performance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.agent_system.phase4 import ClosedLoopAgent, RunStatus, TaskRequest
from src.agent_system.phase4.llm_policy import StructuredLLMPlanningPolicy, StructuredLLMSynthesisPolicy
from src.agent_system.phase4.llm_policy import parse_json_object
from src.agent_system.phase4.orchestrator import SystemVariant, TemplateSynthesisPolicy
from src.agent_system.phase4.schema import ActionPlan, Claim, DraftReport, EvidencePassage
from src.agent_system.phase4.tools import FailingEvidenceTool, FrozenOOFPrognosticTool, StaticEvidenceTool


ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"


def _request() -> TaskRequest:
    frame = pd.read_csv(OOF_PATH)
    row = frame.loc[frame["model"] == "M4_combined_rsf"].iloc[0]
    return TaskRequest(task_id="test-task", case_id=str(row["case_id"]), repeat=int(row["repeat"]))


def _evidence_tool() -> StaticEvidenceTool:
    return StaticEvidenceTool([
        EvidencePassage(
            source_id="TEST_SOURCE",
            passage_id="TEST_P1",
            text="A deterministic evidence passage for Phase 4 testing.",
            metadata={"title": "Test corpus"},
        )
    ])


def _agent(policy=None, evidence=None, variant=SystemVariant.B4_FULL_CLOSED_LOOP):
    return ClosedLoopAgent(
        prognostic_tool=FrozenOOFPrognosticTool(OOF_PATH),
        evidence_tool=evidence or _evidence_tool(),
        synthesis_policy=policy or TemplateSynthesisPolicy(),
        variant=variant,
    )


def test_frozen_oof_tool_returns_one_exact_row():
    request = _request()
    result = FrozenOOFPrognosticTool(OOF_PATH).predict(request)
    assert result.status.value == "SUCCESS"
    assert result.provenance == "PHASE3A_FROZEN_OOF_EVALUATION_ONLY"
    assert result.risk_score is not None


def test_invalid_request_safely_abstains_before_tool_call():
    state = _agent().run(TaskRequest(task_id="bad", case_id="x", repeat=6))
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert [trace.step for trace in state.traces] == ["validate"]


def test_evidence_failure_safely_abstains_without_unsourced_report():
    state = _agent(evidence=FailingEvidenceTool()).run(_request())
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert state.final_report is None
    assert state.abstention_reason == "Simulated retrieval timeout"


def test_verifier_requests_one_revision_and_accepts_corrected_citation():
    state = _agent(policy=TemplateSynthesisPolicy(omit_citation_on_first_attempt=True)).run(_request())
    assert state.status == RunStatus.VERIFIED_REPORT
    assert state.revision_count == 1
    assert [trace.step for trace in state.traces].count("verify") == 2
    assert state.final_report.claims[0].citation_ids == ("TEST_P1",)


def test_verifier_rejects_model_value_mutation():
    class AlteringPolicy:
        def create_draft(self, state, revision):
            altered = replace(state.model_result, risk_score=state.model_result.risk_score + 1.0)
            return DraftReport(
                model_result=altered,
                claims=(Claim("Evidence", ("TEST_P1",)),),
                report_text="Technical report without treatment advice.",
            )

    state = _agent(policy=AlteringPolicy()).run(_request())
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert any(
        finding.code == "MODEL_FIDELITY_FAILURE"
        for finding in state.verification.findings
    )


def test_b3_ablation_emits_unverified_output_without_retry():
    state = _agent(
        policy=TemplateSynthesisPolicy(omit_citation_on_first_attempt=True),
        variant=SystemVariant.B3_MULTI_AGENT_NO_VERIFIER,
    ).run(_request())
    assert state.status == RunStatus.VERIFIED_REPORT
    assert state.revision_count == 0
    assert any(trace.status == "UNVERIFIED" for trace in state.traces)


def test_b2_strong_baseline_does_not_receive_b4_verifier_or_revision():
    state = _agent(
        policy=TemplateSynthesisPolicy(omit_citation_on_first_attempt=True),
        variant=SystemVariant.B2_SINGLE_LLM_WITH_TOOLS,
    ).run(_request())
    assert state.status == RunStatus.VERIFIED_REPORT
    assert state.revision_count == 0
    assert all(trace.step != "verify" for trace in state.traces)
    assert any(trace.status == "UNVERIFIED" for trace in state.traces)


def test_verifier_rejects_paraphrase_not_present_in_cited_passage():
    class ParaphrasingPolicy:
        def create_draft(self, state, revision):
            return DraftReport(
                model_result=state.model_result,
                claims=(Claim("A paraphrase not found in the passage.", ("TEST_P1",)),),
                report_text=TemplateSynthesisPolicy().create_draft(state, revision).report_text,
            )

    state = _agent(policy=ParaphrasingPolicy()).run(_request())
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert any(
        finding.code == "CLAIM_NOT_EXTRACTIVELY_SUPPORTED"
        for finding in state.verification.findings
    )


def test_trace_and_serialized_state_exclude_private_reasoning_fields():
    state = _agent().run(_request())
    rendered = str(state.serializable()).lower()
    assert "chain_of_thought" not in rendered
    assert "private_reasoning" not in rendered


def test_structured_llm_policy_preserves_tool_values_and_valid_citations():
    def completion(system, user):
        assert "JSON only" in system
        assert "TEST_P1" in user
        return '{"claims": [{"text": "A deterministic evidence passage for Phase 4 testing.", "citation_ids": ["TEST_P1"]}], "narrative": "Context supplied."}'

    state = _agent(policy=StructuredLLMSynthesisPolicy(completion)).run(_request())
    assert state.status == RunStatus.VERIFIED_REPORT
    assert state.final_report.model_result == state.model_result
    assert "risk_score=" in state.final_report.report_text


def test_malformed_llm_output_reaches_safe_abstention_after_retry_budget():
    state = _agent(policy=StructuredLLMSynthesisPolicy(lambda system, user: "not json")).run(_request())
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert state.revision_count == 1
    assert any(finding.code == "NO_EVIDENCE_CLAIMS" for finding in state.verification.findings)


def test_invalid_llm_plan_abstains_before_any_tool_call():
    class InvalidPlanner:
        def create_plan(self, request):
            return ActionPlan(
                steps=("evidence_tool", "prognostic_tool"),
                arguments={
                    "case_id": request.case_id,
                    "repeat": request.repeat,
                    "requested_model": request.requested_model,
                },
                status="TEST_INVALID_ORDER",
            )

    agent = ClosedLoopAgent(
        prognostic_tool=FrozenOOFPrognosticTool(OOF_PATH),
        evidence_tool=_evidence_tool(),
        synthesis_policy=TemplateSynthesisPolicy(),
        planning_policy=InvalidPlanner(),
    )
    state = agent.run(_request())
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert [trace.step for trace in state.traces] == ["validate", "plan", "safe_abstain"]


def test_provider_wrapped_json_is_parsed_without_reprompting():
    payload = parse_json_object('Analysis omitted.\n```json\n{"status": "ok"}\n```')
    assert payload == {"status": "ok"}


def test_provider_tool_object_plan_is_normalized_deterministically():
    req = _request()
    payload = {
        "steps": [
            {"tool": "prognostic_tool", "arguments": {
                "case_id": req.case_id, "repeat": req.repeat,
                "requested_model": req.requested_model,
            }},
            {"tool": "evidence_tool", "arguments": {
                "case_id": req.case_id, "repeat": req.repeat,
                "requested_model": req.requested_model,
            }},
        ]
    }
    planner = StructuredLLMPlanningPolicy(lambda system, user: __import__("json").dumps(payload))
    plan = planner.create_plan(req)
    assert plan.steps == ("prognostic_tool", "evidence_tool")
    assert plan.arguments["case_id"] == req.case_id


def test_b4_replans_once_after_invalid_tool_plan():
    class RecoveringPlanner:
        def create_plan(self, request):
            return ActionPlan(steps=(), arguments={}, status="FIRST_INVALID")

        def revise_plan(self, request, finding):
            assert finding == "INVALID_TOOL_PLAN_OR_ARGUMENTS"
            return ActionPlan(
                steps=("prognostic_tool", "evidence_tool"),
                arguments={
                    "case_id": request.case_id,
                    "repeat": request.repeat,
                    "requested_model": request.requested_model,
                },
                status="RECOVERED",
            )

    agent = ClosedLoopAgent(
        prognostic_tool=FrozenOOFPrognosticTool(OOF_PATH),
        evidence_tool=_evidence_tool(),
        synthesis_policy=TemplateSynthesisPolicy(),
        planning_policy=RecoveringPlanner(),
        variant=SystemVariant.B4_FULL_CLOSED_LOOP,
    )
    state = agent.run(_request())
    assert state.status == RunStatus.VERIFIED_REPORT
    assert state.planning_revision_count == 1
    assert [trace.step for trace in state.traces].count("verify_plan") == 2
