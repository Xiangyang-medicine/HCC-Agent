#!/usr/bin/env python3
"""Evaluate already outcome-blind M2T scores against one audited OS file."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prognostic_engine.microarray_transport import bootstrap_external_cindices, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "experiments" / "phase3b" / "microarray_transport" / "GSE116174_GPL570_m2t_scores.csv"
SCORING_AUDIT = ROOT / "experiments" / "phase3b" / "microarray_transport" / "GSE116174_GPL570_SCORING_AUDIT.json"
CLINICAL = ROOT / "data" / "external" / "canonical_microarray" / "GSE116174_GPL570_clinical_os.csv"
OUTPUT = ROOT / "experiments" / "phase3b" / "microarray_transport"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--scoring-audit", type=Path, default=SCORING_AUDIT)
    parser.add_argument("--clinical", type=Path, default=CLINICAL)
    parser.add_argument("--label", default="GSE116174_GPL570")
    args = parser.parse_args()
    scores_path, scoring_audit_path, clinical_path = args.scores.resolve(), args.scoring_audit.resolve(), args.clinical.resolve()
    score_audit = json.loads(scoring_audit_path.read_text(encoding="utf-8"))
    if score_audit.get("outcome_file_opened") is not False or score_audit.get("legacy_files_imported") is not False:
        raise ValueError("Scoring audit does not demonstrate outcome-blind, non-legacy score generation.")
    scores = pd.read_csv(scores_path)
    clinical = pd.read_csv(clinical_path)
    required_scores, required_clinical = {"case_id", "risk_score"}, {"case_id", "survival_months", "event"}
    if required_scores - set(scores.columns) or required_clinical - set(clinical.columns):
        raise ValueError("Required scoring or clinical columns are missing.")
    if scores["case_id"].duplicated().any() or clinical["case_id"].duplicated().any():
        raise ValueError("Case IDs must be unique before external evaluation.")
    merged = clinical.merge(scores, on="case_id", how="inner", validate="one_to_one")
    if len(merged) != len(clinical) or len(merged) != len(scores):
        raise ValueError("Score-to-clinical matching is incomplete; no metric is calculated.")
    times, events, risk = (merged["survival_months"].to_numpy(float), merged["event"].to_numpy(int), merged["risk_score"].to_numpy(float))
    if not np.isfinite(times).all() or not np.isfinite(risk).all() or (times <= 0).any() or not set(events).issubset({0, 1}):
        raise ValueError("Invalid survival or risk values in external evaluation inputs.")
    event_times = times[events.astype(bool)]
    if len(event_times) < 2:
        raise ValueError("Uno C-index is not estimable: fewer than two events.")
    tau = float(np.percentile(event_times, 90))
    result = {
        "status": "COMPLETED_SECONDARY_EXPLORATORY_CROSS_PLATFORM_EVALUATION",
        "claim_boundary": "Validates only the TCGA-derived 15-gene transport model, not RNA-seq M4 or clinical utility.",
        "cohort_platform": args.label,
        "n": int(len(merged)),
        "events": int(events.sum()),
        "score_file_sha256": sha256_file(scores_path),
        "clinical_file_sha256": sha256_file(clinical_path),
        "scoring_audit_sha256": sha256_file(scoring_audit_path),
        "bootstrap": bootstrap_external_cindices(times, events, risk, tau=tau, n_bootstrap=1000, seed=456),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT / f"{args.label}_m2t_evaluation_predictions.csv"
    result_path = OUTPUT / f"{args.label}_M2T_EVALUATION.json"
    merged.to_csv(prediction_path, index=False)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
