"""Generate a fail-closed gate before any Phase 4 formal LLM run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_system.phase4.faults import FaultType


READINESS = ROOT / "experiments" / "phase4" / "readiness"
FORMAL_DIR = ROOT / "experiments" / "phase4" / "formal"

EXPECTED_FAULTS = {
    "INVALID_REQUEST_FIELDS",
    "TRANSIENT_RETRIEVAL_TIMEOUT",
    "PERMANENT_MODEL_FAILURE",
    "MISSING_GENE_FEATURES",
    "MALFORMED_MODEL_OUTPUT",
    "CITATION_METADATA_MISMATCH",
    "CONFLICTING_EVIDENCE",
    "UNSUPPORTED_REQUESTED_CLAIM",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-cost-authorized",
        action="store_true",
        help="Record explicit user authorization for the formal external-token expenditure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow the single interrupted formal JSONL checkpoint after validation by the runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    READINESS.mkdir(parents=True, exist_ok=True)
    evidence_gate_path = READINESS / "PHASE4_FORMAL_EVIDENCE_GATE.json"
    development_gate_path = ROOT / "experiments" / "phase4" / "development_20cases_v1" / "DEVELOPMENT_GATE.json"
    split_path = ROOT / "data" / "phase4_benchmark" / "case_split_manifest.json"
    formal_cases_path = ROOT / "data" / "phase4_benchmark" / "formal_cases_reserved_blinded.json"

    evidence_gate = read_json(evidence_gate_path)
    development_gate = read_json(development_gate_path)
    split_manifest = read_json(split_path)
    formal_cases = read_json(formal_cases_path)

    test_command = [
        sys.executable,
        "-m", "pytest",
        str(ROOT / "tests" / "test_phase4_canonical.py"),
        str(ROOT / "tests" / "test_phase4_benchmark.py"),
        str(ROOT / "tests" / "test_phase4_faults.py"),
        str(ROOT / "tests" / "test_phase4_ablations.py"),
        str(ROOT / "tests" / "test_phase4_formal_runner.py"),
        "-q",
    ]
    test_result = subprocess.run(
        test_command, cwd=ROOT, capture_output=True, text=True, check=False
    )

    canonical_files = [
        ROOT / "src" / "agent_system" / "phase4" / name
        for name in [
            "schema.py", "tools.py", "llm_policy.py", "orchestrator.py",
            "verifier.py", "benchmark.py", "faults.py", "ablations.py",
        ]
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in canonical_files)
    forbidden_imports = [
        token for token in ["llm_agent_evaluation", "agent_evaluator"]
        if token in source_text
    ]
    formal_files = [] if not FORMAL_DIR.exists() else [
        path for path in FORMAL_DIR.rglob("*") if path.is_file()
    ]
    permitted_resume_names = {
        "all_run_records.jsonl",
        "all_RESUME_AUDIT.json",
        "all_case_level_metrics.csv",
        "all_RUN_GATE.json",
    }
    formal_state_valid = (
        not formal_files
        if not args.resume
        else bool(formal_files)
        and {path.name for path in formal_files}.issubset(permitted_resume_names)
        and (FORMAL_DIR / "all_run_records.jsonl").exists()
    )
    implemented_faults = {fault.value for fault in FaultType}
    ablation_path = ROOT / "src" / "agent_system" / "phase4" / "ablations.py"
    formal_runner_path = ROOT / "scripts" / "run_phase4_formal_benchmark.py"

    checks = {
        "development_gate_passed": bool(development_gate.get("success")),
        "formal_evidence_gate_passed": bool(evidence_gate.get("success")),
        "formal_case_count_100": len(formal_cases) == 100,
        "development_formal_overlap_zero": split_manifest.get("overlap_n") == 0,
        "eight_fault_types_implemented": implemented_faults == EXPECTED_FAULTS,
        "phase4_unit_tests_passed": test_result.returncode == 0,
        "no_legacy_evaluator_imports": not forbidden_imports,
        "formal_state_valid_for_requested_mode": formal_state_valid,
        "ablation_harness_ready": ablation_path.exists(),
        "formal_runner_ready": formal_runner_path.exists(),
        "external_token_expenditure_authorized": bool(args.external_cost_authorized),
    }
    failed_checks = [key for key, value in checks.items() if not value]
    if not failed_checks:
        status = "FORMAL_READY"
    elif failed_checks == ["external_token_expenditure_authorized"]:
        status = "FORMAL_READY_PENDING_EXTERNAL_COST_AUTHORIZATION"
    else:
        status = "FORMAL_READINESS_BLOCKED"
    gate = {
        "status": status,
        "success": bool(all(checks.values())),
        "formal_run_permitted": bool(all(checks.values())),
        "checks": checks,
        "implemented_fault_types": sorted(implemented_faults),
        "forbidden_import_findings": forbidden_imports,
        "preexisting_formal_files": [str(path.relative_to(ROOT)) for path in formal_files],
        "phase4_test_exit_code": test_result.returncode,
        "phase4_test_last_line": test_result.stdout.strip().splitlines()[-1] if test_result.stdout.strip() else "",
        "projected_clean_formal_calls": 2040,
        "projected_clean_formal_tokens": 5195640,
        "projected_all_modes_calls_approx": 8736,
        "projected_all_modes_tokens_approx": 22600000,
        "note": "Full formal clean, fault, and ablation execution is projected from development usage; actual usage may differ.",
        "source_hashes_sha256": {path.name: sha256_file(path) for path in canonical_files},
        "asset_hashes_sha256": {
            "formal_passages.jsonl": sha256_file(ROOT / "data" / "phase4_evidence" / "formal_passages.jsonl"),
            "formal_claim_passage_annotations.json": sha256_file(ROOT / "data" / "phase4_evidence" / "formal_claim_passage_annotations.json"),
            "formal_cases_reserved_blinded.json": sha256_file(formal_cases_path),
            "protocol_amendment": sha256_file(ROOT / "docs" / "PHASE4_PROTOCOL_AMENDMENT_V3_1_EXTRACTIVE_EVIDENCE_20260727.md"),
            "resume_protocol": sha256_file(ROOT / "docs" / "PHASE4_FORMAL_INTERRUPTION_AND_RESUME_PROTOCOL_20260727.md"),
            "formal_runner": sha256_file(formal_runner_path),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = READINESS / "PHASE4_FORMAL_READINESS_GATE.json"
    output.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
