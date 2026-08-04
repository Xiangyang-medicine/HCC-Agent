#!/usr/bin/env python3
"""Fit the one frozen TCGA-derived M4 artifact required before Phase 3B scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "prognostic_engine" / "src"))

from prognostic_engine.external_validation import fit_frozen_m4_external_artifact, sha256_file


def main() -> int:
    data_path = ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    output_dir = ROOT / "experiments" / "phase3b" / "derivation"
    artifact_path, manifest_path = fit_frozen_m4_external_artifact(data_path, output_dir)
    print(json.dumps({
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
