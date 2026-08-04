#!/usr/bin/env python3
"""Audit the locally stored GSE14520 source without scoring any model.

The dataset is a microarray cohort and therefore remains exploratory under
Phase 3B amendment v2.  This script never imports or runs legacy result code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "GSE14520"
SUPPLEMENT = DATASET_DIR / "GSE14520_Extra_Supplement.txt"
RAW_ARCHIVE = DATASET_DIR / "GSE14520_RAW.tar"
OUTPUT = PROJECT_ROOT / "experiments" / "phase3b" / "gse14520" / "SOURCE_AUDIT.json"
SOURCE_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    required_columns = {
        "Affy_GSM",
        "Tissue Type",
        "Survival status",
        "Survival months",
        "TNM staging",
        "Age",
    }
    clinical = pd.read_csv(SUPPLEMENT, sep="\t")
    missing_columns = sorted(required_columns - set(clinical.columns))
    tumor = clinical.loc[clinical["Tissue Type"].astype(str) == "Tumor"].copy()
    complete_os = tumor.dropna(subset=["Affy_GSM", "Survival status", "Survival months"])
    cels = sorted(DATASET_DIR.glob("*.CEL.gz"))
    legacy_outputs = sorted(
        path.name
        for path in DATASET_DIR.glob("*")
        if path.name in {
            "all_model_risk_scores.csv",
            "model_comparison_results.csv",
            "figure_km_all_models.pdf",
            "figure_model_comparison.pdf",
            "figure_roc_curves.pdf",
            "figure_score_correlation.pdf",
            "generate_figures.py",
        }
    )

    gates = {
        "official_source_url_recorded": True,
        "supplement_exists": SUPPLEMENT.is_file(),
        "raw_archive_exists": RAW_ARCHIVE.is_file(),
        "clinical_required_columns_present": len(missing_columns) == 0,
        "has_tumor_samples": len(tumor) > 0,
        "has_complete_os_rows": len(complete_os) > 0,
        "microarray_platform_requires_exploratory_label": True,
        "legacy_results_excluded": True,
    }
    result = {
        "status": "EXPLORATORY_SOURCE_AUDITED_NOT_READY_FOR_M4_SCORING",
        "confirmatory_external_validation_ready": False,
        "source": {
            "accession": "GSE14520",
            "url": SOURCE_URL,
            "platforms": ["GPL571", "GPL3921"],
            "analysis_role": "EXPLORATORY_CROSS_PLATFORM_ONLY",
        },
        "local_files": {
            "supplement": str(SUPPLEMENT.relative_to(PROJECT_ROOT)),
            "supplement_sha256": sha256_file(SUPPLEMENT) if SUPPLEMENT.is_file() else None,
            "raw_archive": str(RAW_ARCHIVE.relative_to(PROJECT_ROOT)),
            "raw_archive_sha256": sha256_file(RAW_ARCHIVE) if RAW_ARCHIVE.is_file() else None,
            "cel_file_count": len(cels),
            "first_five_cel_sha256": {
                path.name: sha256_file(path) for path in cels[:5]
            },
        },
        "clinical_audit": {
            "rows_total": int(len(clinical)),
            "tumor_rows": int(len(tumor)),
            "tumor_rows_with_os": int(len(complete_os)),
            "missing_required_columns": missing_columns,
            "outcome_definition": {
                "time_column": "Survival months",
                "event_column": "Survival status",
                "agent_access": "PROHIBITED",
            },
        },
        "legacy_outputs_excluded_from_publication": legacy_outputs,
        "required_before_any_m4_scoring": [
            "Freeze TCGA-derived M4 external-validation artifact and manifest.",
            "Create and test a GPL571/GPL3921 probe-to-15-gene mapping.",
            "Demonstrate a prespecified cross-platform transformation compatible with the frozen model, or abstain from M4 scoring.",
            "Create a new canonical pipeline; do not import generate_figures.py or any listed legacy output.",
        ],
        "gates": {key: bool(value) for key, value in gates.items()},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
