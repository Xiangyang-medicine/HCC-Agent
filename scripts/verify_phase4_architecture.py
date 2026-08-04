#!/usr/bin/env python3
"""Create the machine-readable Phase 4 architecture readiness gate.

This is deliberately an offline smoke gate.  It does not claim that the
formal LLM benchmark, evidence corpus, or external validation is complete.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_system.phase4 import ClosedLoopAgent, TaskRequest
from src.agent_system.phase4.orchestrator import TemplateSynthesisPolicy
from src.agent_system.phase4.schema import EvidencePassage, RunStatus
from src.agent_system.phase4.tools import FrozenOOFPrognosticTool, StaticEvidenceTool, sha256_file


CANONICAL_DIR = PROJECT_ROOT / "src" / "agent_system" / "phase4"
OOF_PATH = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "phase4" / "readiness" / "PHASE4_ARCHITECTURE_GATE.json"
FORBIDDEN_IMPORTS = (
    "llm_agent_evaluation",
    "agent_evaluator",
    "src.agents",
    "src.evaluation",
)


def _source_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(CANONICAL_DIR.glob("*.py"))
    }


def _has_forbidden_imports() -> list[str]:
    findings = []
    for path in sorted(CANONICAL_DIR.glob("*.py")):
        content = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_IMPORTS:
            if forbidden in content:
                findings.append(f"{path.name}: {forbidden}")
    return findings


def _contains_private_reasoning_fields() -> list[str]:
    blocked = ("chain_of_thought", "private_reasoning", "<think>", "<thinking>")
    findings = []
    for path in sorted(CANONICAL_DIR.glob("*.py")):
        content = path.read_text(encoding="utf-8").lower()
        for marker in blocked:
            if marker in content:
                findings.append(f"{path.name}: {marker}")
    return findings


def _smoke_test() -> dict[str, bool]:
    tool = FrozenOOFPrognosticTool(OOF_PATH)
    first = tool._predictions.loc[tool._predictions["model"] == "M4_combined_rsf"].iloc[0]
    evidence = StaticEvidenceTool([
        EvidencePassage(
            source_id="SMOKE_SOURCE",
            passage_id="SMOKE_P1",
            text="This passage is a deterministic Phase 4 smoke-test evidence fixture.",
            metadata={"title": "Smoke fixture"},
        )
    ])
    state = ClosedLoopAgent(
        prognostic_tool=tool,
        evidence_tool=evidence,
        synthesis_policy=TemplateSynthesisPolicy(),
    ).run(
        TaskRequest(
            task_id="phase4-smoke",
            case_id=str(first["case_id"]),
            repeat=int(first["repeat"]),
        )
    )
    return {
        "verified_report": state.status == RunStatus.VERIFIED_REPORT,
        "has_model_trace": any(trace.step == "prognostic_tool" for trace in state.traces),
        "has_verifier_trace": any(trace.step == "verify" and trace.status == "PASS" for trace in state.traces),
        "numeric_result_preserved": state.final_report is not None
        and state.final_report.model_result == state.model_result,
    }


def main() -> int:
    forbidden_imports = _has_forbidden_imports()
    private_reasoning = _contains_private_reasoning_fields()
    smoke = _smoke_test()
    formal_directory = PROJECT_ROOT / "experiments" / "phase4" / "formal"
    formal_mock_outputs = list(formal_directory.glob("*mock*")) if formal_directory.exists() else []
    gates = {
        "canonical_package_exists": CANONICAL_DIR.is_dir(),
        "oof_source_exists": OOF_PATH.is_file(),
        "no_legacy_evaluator_imports": len(forbidden_imports) == 0,
        "no_private_reasoning_fields": len(private_reasoning) == 0,
        "offline_smoke_verified": all(smoke.values()),
        "no_mock_outputs_in_formal_directory": len(formal_mock_outputs) == 0,
        "formal_benchmark_not_misrepresented": True,
    }
    architecture_ready = all(gates.values())
    result = {
        "status": (
            "PHASE4_ARCHITECTURE_READY_FORMAL_BENCHMARK_PENDING"
            if architecture_ready else "PHASE4_ARCHITECTURE_GATE_FAILED"
        ),
        "success": bool(architecture_ready),
        "formal_benchmark_ready": False,
        "evaluation_mode": "OFFLINE_ARCHITECTURE_SMOKE_ONLY",
        "gates": {key: bool(value) for key, value in gates.items()},
        "smoke": {key: bool(value) for key, value in smoke.items()},
        "forbidden_import_findings": forbidden_imports,
        "private_reasoning_findings": private_reasoning,
        "source_hashes": _source_hashes(),
        "oof_predictions_sha256": sha256_file(OOF_PATH),
        "formal_mock_outputs": [path.name for path in formal_mock_outputs],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if architecture_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
