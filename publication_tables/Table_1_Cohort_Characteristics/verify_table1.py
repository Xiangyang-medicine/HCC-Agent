from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data/cohort_characteristics_numeric.csv"
EXCLUDED = ROOT / "source_data/excluded_cohort_record.csv"
SOURCE_GATE = ROOT / "TABLE1_SOURCE_GATE.json"
WORKBOOK = ROOT / "Table_1_Cohort_Characteristics.xlsx"
CSV_OUTPUT = ROOT / "Table_1_Cohort_Characteristics.csv"
PREVIEW = ROOT / "Table_1_Preview.png"
FORMULA_SCAN = ROOT / "TABLE1_FORMULA_ERROR_SCAN.ndjson"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, bool] = {}
    required = [
        SOURCE,
        EXCLUDED,
        SOURCE_GATE,
        WORKBOOK,
        CSV_OUTPUT,
        PREVIEW,
        FORMULA_SCAN,
        ROOT / "TABLE_1_TITLE_AND_NOTES.md",
        ROOT / "build_table1.mjs",
        ROOT / "prepare_table1_source.py",
    ]
    for path in required:
        checks[f"exists_{path.name}"] = path.exists() and path.stat().st_size > 0

    source = pd.read_csv(SOURCE)
    source_gate = json.loads(SOURCE_GATE.read_text(encoding="utf-8"))
    excluded = pd.read_csv(EXCLUDED)
    expected = {
        "TCGA-LIHC": (363, 129),
        "GSE14520": (221, 85),
        "GSE116174": (64, 27),
    }
    checks["source_gate_success"] = source_gate.get("success") is True
    checks["three_analysed_cohorts"] = len(source) == 3
    for cohort, (n, events) in expected.items():
        row = source.loc[source["cohort"] == cohort].iloc[0]
        checks[f"{cohort}_n"] = int(row["patients_n"]) == n
        checks[f"{cohort}_events"] = int(row["events_n"]) == events
        checks[f"{cohort}_stage_sums_to_n"] = (
            int(row["stage_i_n"])
            + int(row["stage_ii_n"])
            + int(row["stage_iii_iv_n"])
            + int(row["stage_missing_n"])
            == n
        )
        checks[f"{cohort}_age_complete"] = int(row["age_available_n"]) == n
        checks[f"{cohort}_sex_complete"] = int(row["sex_available_n"]) == n

    tcga = source.loc[source["cohort"] == "TCGA-LIHC"].iloc[0]
    checks["tcga_grade_sums_to_n"] = (
        int(tcga["grade_g1_g2_n"])
        + int(tcga["grade_g3_g4_n"])
        + int(tcga["grade_missing_n"])
        == 363
    )
    checks["all_15_genes_available"] = bool(
        (source["genes_available_n"] == 15).all()
        and (source["genes_required_n"] == 15).all()
    )
    checks["gpl571_not_analysed"] = (
        len(excluded) == 1
        and excluded.iloc[0]["platform"] == "GPL571"
        and excluded.iloc[0]["analysis_status"] == "NOT_ANALYSED"
        and int(excluded.iloc[0]["complete_os_cases"]) == 21
    )
    checks["formula_scan_clean"] = (
        "matched 0 entries" in FORMULA_SCAN.read_text(encoding="utf-8")
    )
    checks["workbook_minimum_size"] = WORKBOOK.stat().st_size > 10_000

    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    gate = {
        "status": "TABLE1_QA_PASSED" if not failed else "TABLE1_QA_FAILED",
        "success": not failed,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in [WORKBOOK, CSV_OUTPUT, PREVIEW]
        },
    }
    (ROOT / "TABLE1_QA_GATE.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
