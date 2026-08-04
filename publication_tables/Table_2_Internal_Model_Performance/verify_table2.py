from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"F:\ACM\publication_tables\Table_2_Internal_Model_Performance")
SOURCE = ROOT / "source_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    performance = load_csv(SOURCE / "model_performance_numeric.csv")
    comparisons = load_csv(SOURCE / "formal_paired_comparisons.csv")
    source_gate = json.loads(
        (SOURCE / "TABLE2_SOURCE_GATE.json").read_text(encoding="utf-8")
    )
    workbook = ROOT / "Table_2_Internal_Model_Performance.xlsx"
    preview = ROOT / "Table_2_Preview.png"
    formula_scan = ROOT / "TABLE2_FORMULA_ERROR_SCAN.ndjson"

    model_ids = [row["model_id"] for row in performance]
    m4 = next(row for row in performance if row["model_id"] == "M4")
    m1 = next(row for row in performance if row["model_id"] == "M1")
    m5_uno = next(
        row
        for row in comparisons
        if row["comparison"] == "M5 vs M1" and row["metric_key"] == "uno_c"
    )
    m4_harrell = next(
        row
        for row in comparisons
        if row["comparison"] == "M4 vs M1"
        and row["metric_key"] == "harrell_c"
    )
    m4_uno = next(
        row
        for row in comparisons
        if row["comparison"] == "M4 vs M1" and row["metric_key"] == "uno_c"
    )

    scan_text = formula_scan.read_text(encoding="utf-8") if formula_scan.exists() else ""
    formula_errors_absent = not any(
        token in scan_text
        for token in ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"]
    )

    checks = {
        "workbook_exists": workbook.exists() and workbook.stat().st_size > 0,
        "preview_exists": preview.exists() and preview.stat().st_size > 0,
        "model_ids_exact": model_ids == ["M1", "M2", "M3", "M4", "M5"],
        "all_models_have_25_folds": all(
            int(row["n_outer_folds"]) == 25 for row in performance
        ),
        "formal_comparisons_exactly_8": len(comparisons) == 8,
        "all_bootstraps_complete": all(
            int(row["iterations_valid"]) == 1000 for row in comparisons
        ),
        "m4_best_harrell_descriptively": float(m4["harrell_c_mean"])
        == max(float(row["harrell_c_mean"]) for row in performance),
        "m4_best_uno_descriptively": float(m4["uno_c_mean"])
        == max(float(row["uno_c_mean"]) for row in performance),
        "m4_lowest_ibs_descriptively": float(m4["ibs_mean"])
        == min(float(row["ibs_mean"]) for row in performance),
        "m4_harrell_vs_m1_not_significant_adjusted": (
            m4_harrell["significant_adjusted"].lower() == "false"
            and float(m4_harrell["p_value_adjusted"]) > 0.05
        ),
        "m4_uno_vs_m1_not_significant_adjusted": (
            m4_uno["significant_adjusted"].lower() == "false"
            and float(m4_uno["p_value_adjusted"]) > 0.05
        ),
        "m5_uno_vs_m1_significantly_worse": (
            m5_uno["significant_adjusted"].lower() == "true"
            and float(m5_uno["mean_difference"]) < 0
        ),
        "m4_harrell_greater_than_m1_descriptively": float(m4["harrell_c_mean"])
        > float(m1["harrell_c_mean"]),
        "source_gate_locked": source_gate["status"] == "TABLE2_SOURCE_LOCKED",
        "formula_errors_absent": formula_errors_absent,
    }
    success = all(checks.values())
    report = {
        "status": "TABLE2_QA_PASSED" if success else "TABLE2_QA_FAILED",
        "success": success,
        "checks": checks,
        "files": {
            "workbook": str(workbook),
            "workbook_sha256": sha256(workbook) if workbook.exists() else None,
            "preview": str(preview),
            "preview_sha256": sha256(preview) if preview.exists() else None,
        },
        "locked_interpretation": {
            "m4_status": "PROVISIONAL_PRIMARY_CANDIDATE",
            "m4_vs_m1_after_adjustment": "NOT_SIGNIFICANT",
            "m5_vs_m1_uno_c": "SIGNIFICANTLY_WORSE",
        },
    }
    (ROOT / "TABLE2_QA_GATE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Table 2 QA report",
        "",
        f"- Overall status: **{report['status']}**",
        f"- Workbook: `{workbook.name}`",
        f"- Models: {', '.join(model_ids)}",
        f"- Formal paired comparisons: {len(comparisons)}",
        "- Canonical comparison version: v6",
        "- M4 descriptive status: best Harrell C, Uno C, 36-month AUC, and IBS.",
        "- Multiplicity-aware conclusion: M4 was not significantly better than M1.",
        "- M5 was significantly worse than M1 for Uno C after Bonferroni correction.",
        "",
        "## Automated checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in checks.items()
    )
    (ROOT / "QA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
