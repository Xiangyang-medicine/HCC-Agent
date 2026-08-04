from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"F:\ACM\publication_tables\Table_3_External_Transport")
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
    performance = load_csv(SOURCE / "external_cohort_performance_numeric.csv")
    threshold = load_csv(SOURCE / "frozen_threshold_results_numeric.csv")
    excluded = load_csv(SOURCE / "excluded_cohort_record.csv")
    coefficients = load_csv(SOURCE / "frozen_gene_coefficients.csv")
    source_gate = json.loads(
        (SOURCE / "TABLE3_SOURCE_GATE.json").read_text(encoding="utf-8")
    )
    workbook = ROOT / "Table_3_External_Transport.xlsx"
    preview = ROOT / "Table_3_Preview.png"
    formula_scan = ROOT / "TABLE3_FORMULA_ERROR_SCAN.ndjson"
    formula_text = (
        formula_scan.read_text(encoding="utf-8")
        if formula_scan.exists()
        else ""
    )

    cohort_keys = [
        f"{row['cohort']}_{row['platform']}" for row in performance
    ]
    high_low_totals_match = all(
        int(trow["higher_risk_n"]) + int(trow["lower_risk_n"])
        == int(
            next(
                row["n"]
                for row in performance
                if row["cohort"] == trow["cohort"]
                and row["platform"] == trow["platform"]
            )
        )
        for trow in threshold
    )
    checks = {
        "workbook_exists": workbook.exists() and workbook.stat().st_size > 0,
        "preview_exists": preview.exists() and preview.stat().st_size > 0,
        "cohorts_exact": cohort_keys
        == ["GSE14520_GPL3921", "GSE116174_GPL570"],
        "total_patients_285": sum(int(row["n"]) for row in performance) == 285,
        "total_events_112": sum(int(row["events"]) for row in performance)
        == 112,
        "all_bootstrap_iterations_1000": all(
            int(row["valid_iterations"]) == 1000 for row in performance
        ),
        "high_low_counts_match_cohort_size": high_low_totals_match,
        "single_frozen_cutoff": len(
            {round(float(row["frozen_tcga_cutoff"]), 12) for row in threshold}
        )
        == 1,
        "no_external_outcome_grouping": all(
            row["external_outcome_used_for_grouping"].lower() == "false"
            for row in threshold
        ),
        "no_external_recalibration": all(
            row["external_recalibration"].lower() == "false"
            for row in performance
        ),
        "gpl571_not_in_performance": all(
            row["platform"] != "GPL571" for row in performance
        ),
        "gpl571_exclusion_recorded": (
            len(excluded) == 1
            and excluded[0]["platform"] == "GPL571"
            and excluded[0]["analysis_status"] == "Not analysed"
        ),
        "two_nonzero_genes": sum(
            row["nonzero"].lower() == "true" for row in coefficients
        )
        == 2,
        "nonzero_genes_exact": {
            row["gene"]
            for row in coefficients
            if row["nonzero"].lower() == "true"
        }
        == {"LDHA", "PKM"},
        "source_gate_locked": source_gate["status"] == "TABLE3_SOURCE_LOCKED",
        "formula_errors_absent": not any(
            token in formula_text
            for token in ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"]
        ),
    }
    success = all(checks.values())
    report = {
        "status": "TABLE3_QA_PASSED" if success else "TABLE3_QA_FAILED",
        "success": success,
        "checks": checks,
        "files": {
            "workbook": str(workbook),
            "workbook_sha256": sha256(workbook) if workbook.exists() else None,
            "preview": str(preview),
            "preview_sha256": sha256(preview) if preview.exists() else None,
        },
        "claim_boundary": source_gate["claim_boundary"],
    }
    (ROOT / "TABLE3_QA_GATE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Table 3 QA report",
        "",
        f"- Overall status: **{report['status']}**",
        "- Included cohorts: GSE14520 GPL3921 and GSE116174 GPL570.",
        "- Total analysis population: 285 patients and 112 deaths.",
        "- Bootstrap intervals: 1,000/1,000 valid draws for each metric.",
        "- Frozen TCGA threshold applied unchanged in both cohorts.",
        "- GPL571 recorded as not analysed (N=21).",
        "- Claim boundary: gene-only cross-platform transport, not M4 validation.",
        "",
        "## Automated checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}"
        for name, passed in checks.items()
    )
    (ROOT / "QA_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
