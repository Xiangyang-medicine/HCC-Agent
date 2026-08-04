#!/usr/bin/env python3
"""Verify integrity and smoke-score behavior of the frozen Phase 3B M4 artifact."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "prognostic_engine" / "src"))

from prognostic_engine.external_validation import ARTIFACT_STATUS, MODEL_ID, sha256_file


DERIVATION_DIR = ROOT / "experiments" / "phase3b" / "derivation"
ARTIFACT_PATH = DERIVATION_DIR / "m4_external_validation_artifact.joblib"
MANIFEST_PATH = DERIVATION_DIR / "M4_EXTERNAL_VALIDATION_MANIFEST.json"
DATA_PATH = ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
OUTPUT = ROOT / "experiments" / "phase3b" / "derivation" / "ARTIFACT_VERIFICATION.json"


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact = joblib.load(ARTIFACT_PATH)
    data = pd.read_parquet(DATA_PATH)
    required = ["case_id", "age_at_diagnosis", "ajcc_stage", "tumor_grade", *artifact.transformer.gene_columns]
    smoke_input = data.loc[:, required].dropna().head(5)
    smoke_output = artifact.predict(smoke_input)
    gates = {
        "artifact_manifest_hash_matches": manifest.get("artifact_sha256") == sha256_file(ARTIFACT_PATH),
        "derivation_data_hash_matches": manifest.get("derivation_data_sha256") == sha256_file(DATA_PATH),
        "model_id_matches": artifact.model_id == MODEL_ID and manifest.get("model_id") == MODEL_ID,
        "artifact_is_not_clinical_deployment": artifact.artifact_status == ARTIFACT_STATUS,
        "manifest_reports_363_patients_129_events": manifest.get("derivation_n") == 363 and manifest.get("derivation_events") == 129,
        "external_outcomes_not_used_for_fitting": manifest.get("external_input_policy", {}).get("external_outcomes_used_for_fitting") is False,
        "smoke_input_feature_complete": len(smoke_input) == 5,
        "smoke_predictions_finite": bool(np.isfinite(smoke_output.drop(columns=["case_id"]).to_numpy()).all()),
        "smoke_prediction_has_expected_columns": set(smoke_output.columns) == {
            "case_id", "risk_score", "survival_probability_12m", "survival_probability_36m", "survival_probability_60m"
        },
    }
    result = {
        "status": "PHASE3B_DERIVATION_ARTIFACT_VERIFIED" if all(gates.values()) else "PHASE3B_DERIVATION_ARTIFACT_FAILED",
        "success": bool(all(gates.values())),
        "gates": {key: bool(value) for key, value in gates.items()},
        "artifact_sha256": sha256_file(ARTIFACT_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "smoke_case_count": int(len(smoke_output)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
