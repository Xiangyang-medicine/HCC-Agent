from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"F:\ACM\publication_tables\Table_4_Agent_Benchmark")
SOURCE = ROOT / "source_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    primary = read_csv(SOURCE / "primary_endpoint_numeric.csv")
    paired = read_csv(SOURCE / "primary_paired_comparison_numeric.csv")[0]
    trace = read_csv(SOURCE / "traceability_reliability_numeric.csv")
    planning = read_csv(SOURCE / "planning_audit_numeric.csv")
    ablations = read_csv(SOURCE / "ablation_effects_numeric.csv")
    faults = read_csv(SOURCE / "fault_handling_numeric.csv")
    systems = read_csv(SOURCE / "system_definitions.csv")
    source_gate = json.loads(
        (SOURCE / "TABLE4_SOURCE_GATE.json").read_text(encoding="utf-8")
    )
    workbook = ROOT / "Table_4_Agent_Benchmark.xlsx"
    preview = ROOT / "Table_4_Preview.png"
    formula_scan = ROOT / "TABLE4_FORMULA_ERROR_SCAN.ndjson"
    formula_text = (
        formula_scan.read_text(encoding="utf-8")
        if formula_scan.exists()
        else ""
    )

    b2 = next(row for row in primary if row["system"].startswith("B2_"))
    b4 = next(row for row in primary if row["system"].startswith("B4_"))
    unsupported = next(
        row for row in faults if row["fault_type"] == "UNSUPPORTED_REQUESTED_CLAIM"
    )

    checks = {
        "workbook_exists": workbook.exists() and workbook.stat().st_size > 0,
        "preview_exists": preview.exists() and preview.stat().st_size > 0,
        "formal_record_count_4860": source_gate["record_count"] == 4860,
        "formal_unique_count_4860": source_gate["unique_record_count"] == 4860,
        "no_api_errors": source_gate["api_error_count"] == 0,
        "systems_exact_b2_b4": [row["system"] for row in systems]
        == ["B2_SINGLE_LLM_WITH_TOOLS", "B4_FULL_CLOSED_LOOP"],
        "primary_300_runs_each": all(int(row["n_runs"]) == 300 for row in primary),
        "primary_success_counts": int(b2["successes"]) == 245
        and int(b4["successes"]) == 284,
        "primary_rates": abs(float(b2["rate"]) - 0.8166666667) < 1e-9
        and abs(float(b4["rate"]) - 0.9466666667) < 1e-9,
        "primary_difference_13pp": abs(float(paired["difference"]) - 0.13) < 1e-12,
        "primary_difference_ci": abs(float(paired["ci_lower"]) - 0.08) < 1e-12
        and abs(float(paired["ci_upper"]) - 0.1833333333333333) < 1e-12,
        "paired_outcomes_sum_300": sum(
            int(paired[key])
            for key in ["both_pass", "b4_only_pass", "b2_only_pass", "both_fail"]
        )
        == 300,
        "traceability_rows_6": len(trace) == 6,
        "planning_rows_2": len(planning) == 2,
        "four_ablations": len(ablations) == 4,
        "eight_fault_types": len(faults) == 8,
        "unsupported_scoring_mismatch_preserved": bool(
            unsupported["scoring_note"].strip()
        ),
        "confirmatory_endpoint_unchanged": source_gate[
            "confirmatory_endpoint_modified"
        ]
        is False,
        "strict_audit_zero_disagreement": source_gate[
            "strict_vs_frozen_clean_disagreement_count"
        ]
        == 0,
        "schema_metric_absent": source_gate["schema_metric_in_main_table"] is False,
        "b2_verifier_not_zero_coded": source_gate[
            "b2_internal_verifier_encoded_as_zero"
        ]
        is False,
        "source_gate_locked": source_gate["status"] == "TABLE4_SOURCE_LOCKED",
        "formula_errors_absent": not any(
            token in formula_text
            for token in ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"]
        ),
    }
    success = all(checks.values())
    report = {
        "status": "TABLE4_QA_PASSED" if success else "TABLE4_QA_FAILED",
        "success": success,
        "checks": checks,
        "files": {
            "workbook": str(workbook),
            "workbook_sha256": sha256(workbook) if workbook.exists() else None,
            "preview": str(preview),
            "preview_sha256": sha256(preview) if preview.exists() else None,
        },
        "locked_primary_result": {
            "B2": "245/300 (81.7%)",
            "B4": "284/300 (94.7%)",
            "absolute_difference": "+13.0 percentage points",
            "adjusted_or_primary_p": float(paired["p_value"]),
        },
        "claim_boundary": source_gate["claim_boundary"],
    }
    (ROOT / "TABLE4_QA_GATE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Table 4 QA report",
        "",
        f"- Overall status: **{report['status']}**",
        "- Formal records: 4,860/4,860 unique; 0 API errors.",
        "- Primary B2 result: 245/300 (81.7%).",
        "- Primary B4 result: 284/300 (94.7%).",
        "- Primary difference: +13.0 percentage points.",
        "- Strict post-hoc report-contract audit: 0/600 clean-run disagreements.",
        "- Main table excludes the misleading `schema_valid` label.",
        "- B2 internal verifier is N/A, not zero.",
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
