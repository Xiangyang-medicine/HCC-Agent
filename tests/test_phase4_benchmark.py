from pathlib import Path

import pandas as pd

from src.agent_system.phase4.benchmark import run_comparator, score_clean_run
from src.agent_system.phase4.orchestrator import SystemVariant, TemplateSynthesisPolicy, ClosedLoopAgent
from src.agent_system.phase4.schema import EvidencePassage, TaskRequest
from src.agent_system.phase4.tools import FrozenOOFPrognosticTool, StaticEvidenceTool


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"


def request():
    frame = pd.read_csv(OOF)
    row = frame.loc[frame["model"] == "M4_combined_rsf"].iloc[0]
    return TaskRequest("benchmark-test", str(row["case_id"]), int(row["repeat"]))


def evidence():
    return StaticEvidenceTool([
        EvidencePassage(
            source_id="S1",
            passage_id="P1",
            text="This exact evidence sentence is available for extraction.",
            metadata={"title": "Test"},
        )
    ])


def test_b4_template_run_passes_external_clean_scorer():
    req = request()
    tool = FrozenOOFPrognosticTool(OOF)
    ev = evidence()
    agent = ClosedLoopAgent(
        prognostic_tool=tool,
        evidence_tool=ev,
        synthesis_policy=TemplateSynthesisPolicy(),
        variant=SystemVariant.B4_FULL_CLOSED_LOOP,
    )
    state = agent.run(req)
    score = score_clean_run(state, SystemVariant.B4_FULL_CLOSED_LOOP, req, tool, ev)
    assert score["verified_task_success"] is True
    assert score["numeric_fidelity"] is True


def test_b2_missing_citation_fails_external_clean_scorer():
    req = request()
    tool = FrozenOOFPrognosticTool(OOF)
    ev = evidence()
    agent = ClosedLoopAgent(
        prognostic_tool=tool,
        evidence_tool=ev,
        synthesis_policy=TemplateSynthesisPolicy(omit_citation_on_first_attempt=True),
        variant=SystemVariant.B2_SINGLE_LLM_WITH_TOOLS,
    )
    state = agent.run(req)
    score = score_clean_run(state, SystemVariant.B2_SINGLE_LLM_WITH_TOOLS, req, tool, ev)
    assert score["verified_task_success"] is False
    assert score["citation_completeness"] == 0.0


def test_b1_no_tools_cannot_pass_core_task():
    req = request()
    tool = FrozenOOFPrognosticTool(OOF)
    ev = evidence()
    completion = lambda system, user: '{"claims": [], "narrative": "No evidence."}'
    state = run_comparator(req, SystemVariant.B1_SINGLE_LLM_NO_TOOLS, tool, ev, completion)
    score = score_clean_run(state, SystemVariant.B1_SINGLE_LLM_NO_TOOLS, req, tool, ev)
    assert score["verified_task_success"] is False
    assert score["tool_selection_recall"] == 0.0
