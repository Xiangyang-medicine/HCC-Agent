from pathlib import Path

import pandas as pd

from src.agent_system.phase4.faults import (
    ConflictingEvidenceTool,
    FaultType,
    MetadataMismatchEvidenceTool,
    NonfiniteModelOutputTool,
    PermanentFailingPrognosticTool,
    TransientEvidenceTool,
    invalid_request_fields,
    score_fault_run,
)
from src.agent_system.phase4.orchestrator import ClosedLoopAgent, SystemVariant, TemplateSynthesisPolicy
from src.agent_system.phase4.schema import EvidencePassage, RunStatus, TaskRequest
from src.agent_system.phase4.tools import FrozenOOFPrognosticTool, StaticEvidenceTool


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"


def request():
    frame = pd.read_csv(OOF)
    row = frame.loc[frame["model"] == "M4_combined_rsf"].iloc[0]
    return TaskRequest("fault-test", str(row["case_id"]), int(row["repeat"]))


def evidence():
    return StaticEvidenceTool([
        EvidencePassage(
            source_id="PMID_1",
            passage_id="PMID_1_P01",
            text="This exact evidence sentence is available for extraction.",
            metadata={"pmid": "1", "source_url": "https://pubmed.ncbi.nlm.nih.gov/1/"},
        )
    ])


def b4(prognostic, evidence_tool):
    return ClosedLoopAgent(
        prognostic_tool=prognostic,
        evidence_tool=evidence_tool,
        synthesis_policy=TemplateSynthesisPolicy(),
        variant=SystemVariant.B4_FULL_CLOSED_LOOP,
    )


def test_b4_recovers_from_one_transient_retrieval_timeout():
    req = request()
    prognostic = FrozenOOFPrognosticTool(OOF)
    clean_evidence = evidence()
    state = b4(prognostic, TransientEvidenceTool(clean_evidence)).run(req)
    score = score_fault_run(
        state, FaultType.TRANSIENT_RETRIEVAL_TIMEOUT,
        SystemVariant.B4_FULL_CLOSED_LOOP, req, prognostic, clean_evidence,
    )
    assert state.status == RunStatus.VERIFIED_REPORT
    assert score["failure_detected"] is True
    assert score["recovery_or_safe_abstention"] is True


def test_b4_safely_abstains_on_invalid_request_fields():
    req = invalid_request_fields(request())
    prognostic = FrozenOOFPrognosticTool(OOF)
    clean_evidence = evidence()
    state = b4(prognostic, clean_evidence).run(req)
    score = score_fault_run(
        state, FaultType.INVALID_REQUEST_FIELDS,
        SystemVariant.B4_FULL_CLOSED_LOOP, req, prognostic, clean_evidence,
    )
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert score["failure_detected"] is True
    assert score["recovery_or_safe_abstention"] is True


def test_b4_safely_abstains_on_permanent_model_failure():
    req = request()
    state = b4(PermanentFailingPrognosticTool(), evidence()).run(req)
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert state.final_report is None


def test_b4_rejects_nonfinite_model_output():
    req = request()
    state = b4(NonfiniteModelOutputTool(FrozenOOFPrognosticTool(OOF)), evidence()).run(req)
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert any(f.code == "MODEL_RESULT_NONFINITE" for f in state.verification.findings)


def test_b4_rejects_evidence_metadata_mismatch():
    req = request()
    state = b4(
        FrozenOOFPrognosticTool(OOF), MetadataMismatchEvidenceTool(evidence())
    ).run(req)
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert any(f.code == "EVIDENCE_METADATA_MISMATCH" for f in state.verification.findings)


def test_b4_rejects_unresolved_conflicting_evidence():
    req = request()
    state = b4(
        FrozenOOFPrognosticTool(OOF), ConflictingEvidenceTool(evidence())
    ).run(req)
    assert state.status == RunStatus.SAFE_ABSTAIN
    assert any(f.code == "CONFLICTING_EVIDENCE_UNRESOLVED" for f in state.verification.findings)
