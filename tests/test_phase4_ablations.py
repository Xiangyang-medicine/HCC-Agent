from pathlib import Path

import pandas as pd

from src.agent_system.phase4.ablations import AblationVariant, StatelessLLMSynthesisPolicy, build_ablation_agent
from src.agent_system.phase4.orchestrator import SystemVariant
from src.agent_system.phase4.schema import EvidencePassage, RunState, TaskRequest
from src.agent_system.phase4.tools import FrozenOOFPrognosticTool, StaticEvidenceTool
from src.agent_system.phase4.verifier import ReportVerifier


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"


def request():
    frame = pd.read_csv(OOF)
    row = frame.loc[frame["model"] == "M4_combined_rsf"].iloc[0]
    return TaskRequest("ablation-test", str(row["case_id"]), int(row["repeat"]))


def evidence():
    return StaticEvidenceTool([
        EvidencePassage("S1", "P1", "Exact evidence sentence.", {"title": "Test"})
    ])


def planning_json(req):
    return {
        "steps": ["prognostic_tool", "evidence_tool"],
        "arguments": {
            "case_id": req.case_id,
            "repeat": req.repeat,
            "requested_model": req.requested_model,
        },
    }


def test_all_four_ablation_variants_construct():
    req = request()
    import json

    def completion(system, user):
        if "planning component" in system:
            return json.dumps(planning_json(req))
        return json.dumps({
            "claims": [{"text": "Exact evidence sentence.", "citation_ids": ["P1"]}],
            "narrative": "Context.",
            "model_id": "M4_combined_rsf",
            "risk_score": 0,
            "survival_probability_36m": 0,
        })

    for variant in AblationVariant:
        agent = build_ablation_agent(
            variant, FrozenOOFPrognosticTool(OOF), evidence(), completion
        )
        assert agent is not None


def test_no_evidence_contract_flag_is_disabled_only_for_that_ablation():
    req = request()
    import json

    def completion(system, user):
        if "planning component" in system:
            return json.dumps(planning_json(req))
        return json.dumps({
            "claims": [{"text": "Paraphrase.", "citation_ids": ["P1"]}],
            "narrative": "Context.",
        })

    agent = build_ablation_agent(
        AblationVariant.NO_EVIDENCE_CONTRACT,
        FrozenOOFPrognosticTool(OOF), evidence(), completion,
    )
    assert agent.verifier.enforce_extractive_support is False


def test_no_revision_loop_disables_all_retries():
    req = request()
    agent = build_ablation_agent(
        AblationVariant.NO_REVISION_LOOP,
        FrozenOOFPrognosticTool(OOF), evidence(), lambda system, user: "{}",
    )
    state = agent.run(req)
    assert state.planning_revision_count == 0
    assert state.revision_count == 0
