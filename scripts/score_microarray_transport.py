#!/usr/bin/env python3
"""Generate M2T scores from outcome-free canonical microarray gene inputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prognostic_engine.config import METABOLIC_GENES
from prognostic_engine.microarray_transport import external_cohort_zscore, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "external" / "canonical_microarray" / "GSE116174_GPL570_15gene_unscored.csv"
DEFAULT_ARTIFACT = ROOT / "experiments" / "phase3b" / "derivation" / "m2t_crossplatform_artifact.joblib"
OUTPUT_DIR = ROOT / "experiments" / "phase3b" / "microarray_transport"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--label", default="GSE116174_GPL570")
    args = parser.parse_args()
    input_path, artifact_path = args.input.resolve(), args.artifact.resolve()
    if "risk" in input_path.name.lower() or "validation" in input_path.name.lower():
        raise ValueError("Canonical scoring input must be the unscored gene matrix, never a legacy result file.")
    genes = pd.read_csv(input_path, index_col=0)
    if list(genes.index) != METABOLIC_GENES:
        raise ValueError("Input gene order does not match the locked 15-gene panel.")
    artifact = joblib.load(artifact_path)
    if artifact.artifact_status != "EXPLORATORY_EXTERNAL_TRANSPORT_ONLY_NOT_FOR_CLINICAL_DEPLOYMENT":
        raise ValueError("Invalid M2T artifact status.")
    # This call uses expression only. The script intentionally does not open an outcome file.
    scores = artifact.predict(external_cohort_zscore(genes))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.label}_m2t_scores.csv"
    audit_path = OUTPUT_DIR / f"{args.label}_SCORING_AUDIT.json"
    scores.to_csv(output_path, index=False)
    audit = {
        "status": "SCORED_OUTCOME_BLIND_AWAITING_SEPARATE_EVALUATION",
        "cohort_platform": args.label,
        "input_path": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256_file(input_path),
        "artifact_path": str(artifact_path.relative_to(ROOT)),
        "artifact_sha256": sha256_file(artifact_path),
        "risk_score_count": int(len(scores)),
        "risk_scores_finite": bool(np.isfinite(scores["risk_score"].to_numpy(dtype=float)).all()),
        "outcome_file_opened": False,
        "legacy_files_imported": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
