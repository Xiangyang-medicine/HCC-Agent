#!/usr/bin/env python3
"""Verify the complete secondary microarray transport analysis and write a gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prognostic_engine.microarray_transport import sha256_file


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "phase3b" / "microarray_transport"
ARTIFACT = ROOT / "experiments" / "phase3b" / "derivation" / "m2t_crossplatform_artifact.joblib"

EXPECTED = {
    "GSE116174_GPL570": {"n": 64, "events": 27},
    "GSE14520_GPL571": {"n": 21, "events": 11},
    "GSE14520_GPL3921": {"n": 221, "events": 85},
}


def main() -> int:
    audit = json.loads((BASE / "MICROARRAY_SOURCE_AND_INPUT_AUDIT.json").read_text(encoding="utf-8"))
    artifact = joblib.load(ARTIFACT)
    checks: dict[str, bool] = {
        "protocol_scope_is_secondary_exploratory": audit.get("analysis_scope") == "SECONDARY_EXPLORATORY_CROSS_PLATFORM_GENE_LAYER_ONLY",
        "legacy_files_not_imported": audit.get("legacy_files_imported") is False,
        "m2t_artifact_status_is_nonclinical": artifact.artifact_status == "EXPLORATORY_EXTERNAL_TRANSPORT_ONLY_NOT_FOR_CLINICAL_DEPLOYMENT",
    }
    strata = {}
    for label, expected in EXPECTED.items():
        score_path = BASE / f"{label}_m2t_scores.csv"
        scoring_audit_path = BASE / f"{label}_SCORING_AUDIT.json"
        evaluation_path = BASE / f"{label}_M2T_EVALUATION.json"
        scores = pd.read_csv(score_path)
        scoring_audit = json.loads(scoring_audit_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        local_checks = {
            "expected_n": len(scores) == expected["n"] == evaluation.get("n"),
            "expected_events": evaluation.get("events") == expected["events"],
            "unique_case_id": not scores["case_id"].duplicated().any(),
            "finite_risk": bool(np.isfinite(scores["risk_score"].to_numpy(dtype=float)).all()),
            "outcome_blind_scoring": scoring_audit.get("outcome_file_opened") is False,
            "legacy_excluded_at_scoring": scoring_audit.get("legacy_files_imported") is False,
            "artifact_hash_match": scoring_audit.get("artifact_sha256") == sha256_file(ARTIFACT),
            "bootstrap_complete": evaluation.get("bootstrap", {}).get("valid_iterations") == 1000,
        }
        checks.update({f"{label}_{name}": bool(value) for name, value in local_checks.items()})
        strata[label] = {
            "checks": local_checks,
            "harrell_c": evaluation["bootstrap"]["metrics"]["harrell_c"],
            "uno_c": evaluation["bootstrap"]["metrics"]["uno_c"],
        }
    result = {
        "status": "COMPLETED_SECONDARY_EXPLORATORY_TRANSPORT_ANALYSIS" if all(checks.values()) else "FAILED_INTEGRITY_GATE",
        "success": bool(all(checks.values())),
        "checks": checks,
        "artifact_sha256": sha256_file(ARTIFACT),
        "strata": strata,
        "reporting_rule": "Report all three platform strata. Do not pool patients, claim M4 validation, or omit the GPL571 result.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = BASE / "MICROARRAY_TRANSPORT_GATE.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
