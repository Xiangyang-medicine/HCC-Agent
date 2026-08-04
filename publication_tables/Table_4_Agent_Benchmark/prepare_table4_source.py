from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(r"F:\ACM\publication_tables\Table_4_Agent_Benchmark")
SOURCE_DIR = ROOT / "source_data"
ACM = Path(r"F:\ACM")
FIGURE = ACM / "publication_figures" / "Figure_4_Agent_Benchmark"
FORMAL = ACM / "experiments" / "phase4" / "formal_remote_completed"

SYSTEM_PATH = FIGURE / "panel_a_system_contrast" / "source_data.csv"
PRIMARY_PATH = FIGURE / "panel_b_primary_endpoint" / "source_data.csv"
PAIRED_PATH = (
    FIGURE / "panel_b_primary_endpoint" / "paired_comparison.csv"
)
ABLATION_PATH = FIGURE / "panel_c_ablation_effects" / "source_data.csv"
TRACEABILITY_PATH = (
    FIGURE / "panel_d_traceability_reliability" / "source_data.csv"
)
FAULT_PATH = FIGURE / "panel_e_fault_handling" / "source_data.csv"
PLANNING_PATH = (
    FIGURE / "methodological_audit" / "planning_error_summary.csv"
)
CORRECTIONS_PATH = (
    FIGURE
    / "methodological_audit"
    / "metric_definition_corrections.csv"
)
AUDIT_GATE_PATH = FIGURE / "methodological_audit" / "AUDIT_GATE.json"
FIGURE_QA_PATH = FIGURE / "FIGURE4_QA_GATE.json"
FORMAL_GATE_PATH = FORMAL / "all_RUN_GATE.json"
FORMAL_RECORDS_PATH = FORMAL / "all_run_records.jsonl"

SYSTEM_ORDER = [
    "B2_SINGLE_LLM_WITH_TOOLS",
    "B4_FULL_CLOSED_LOOP",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value}")
    return result


def p_display(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    systems_source = read_csv(SYSTEM_PATH)
    primary_source = read_csv(PRIMARY_PATH)
    paired_source = read_csv(PAIRED_PATH)
    ablation_source = read_csv(ABLATION_PATH)
    trace_source = read_csv(TRACEABILITY_PATH)
    fault_source = read_csv(FAULT_PATH)
    planning_source = read_csv(PLANNING_PATH)
    corrections_source = read_csv(CORRECTIONS_PATH)
    audit_gate = json.loads(AUDIT_GATE_PATH.read_text(encoding="utf-8-sig"))
    figure_qa = json.loads(FIGURE_QA_PATH.read_text(encoding="utf-8-sig"))
    formal_gate = json.loads(FORMAL_GATE_PATH.read_text(encoding="utf-8-sig"))

    if figure_qa.get("success") is not True:
        raise ValueError("Figure 4 QA gate did not pass.")
    if formal_gate.get("record_count") != 4860:
        raise ValueError("Formal benchmark record count is not 4,860.")
    if formal_gate.get("api_error_count") != 0:
        raise ValueError("Formal benchmark contains API errors.")
    if audit_gate.get("confirmatory_endpoint_modified") is not False:
        raise ValueError("Confirmatory endpoint was modified.")

    systems = {
        row["system"]: {
            "system": row["system"],
            "display_name": row["display_name"],
            "controller": row["controller"],
            "tools": row["tools"],
            "internal_verifier": row["internal_verifier"],
            "conditional_repair": row["conditional_repair"],
            "terminal_states": row["terminal_states"],
        }
        for row in systems_source
        if row["system"] in SYSTEM_ORDER
    }
    system_rows = [systems[key] for key in SYSTEM_ORDER]

    primary_rows = []
    for system in SYSTEM_ORDER:
        row = next(
            item
            for item in primary_source
            if item["system"] == system
            and item["metric"] == "frozen_external_composite_pass"
        )
        primary_rows.append(
            {
                "system": system,
                "display_name": row["display_name"],
                "endpoint": row["metric"],
                "endpoint_label": row["metric_label"],
                "analysis_status": row["analysis_status"],
                "n_cases": int(row["n_cases"]),
                "n_runs": int(row["n_runs"]),
                "successes": int(row["successes"]),
                "rate": as_float(row["rate"]),
                "ci_lower": as_float(row["ci_low"]),
                "ci_upper": as_float(row["ci_high"]),
            }
        )

    pair_raw = paired_source[0]
    paired = {
        "comparison": "B4 vs B2",
        "effect_definition": "B4 minus B2",
        "difference": as_float(pair_raw["difference"]),
        "ci_lower": as_float(pair_raw["ci_low"]),
        "ci_upper": as_float(pair_raw["ci_high"]),
        "p_value": as_float(pair_raw["p_value"]),
        "n_cases": int(pair_raw["n_cases"]),
        "n_bootstrap": int(pair_raw["n_bootstrap"]),
        "n_permutation": int(pair_raw["n_permutation"]),
        "both_pass": int(pair_raw["both_pass"]),
        "b4_only_pass": int(pair_raw["b4_only_pass"]),
        "b2_only_pass": int(pair_raw["b2_only_pass"]),
        "both_fail": int(pair_raw["both_fail"]),
        "n_paired_runs": int(pair_raw["n_paired_runs"]),
    }

    trace_rows = []
    for metric in [
        "exact_extractive_claim_support",
        "retrieved_passage_citation_validity",
        "exact_three_run_agreement",
    ]:
        for system in SYSTEM_ORDER:
            row = next(
                item
                for item in trace_source
                if item["system"] == system and item["metric"] == metric
            )
            trace_rows.append(
                {
                    "system": system,
                    "display_name": row["display_name"],
                    "metric": metric,
                    "metric_label": row["metric_label"],
                    "reference_standard": row["reference_standard"],
                    "n_cases": int(row["n_cases"]),
                    "n_runs": int(row["n_runs"]),
                    "value": as_float(row["value"]),
                    "ci_lower": as_float(row["ci_low"]),
                    "ci_upper": as_float(row["ci_high"]),
                }
            )

    planning_rows = []
    for system in SYSTEM_ORDER:
        row = next(item for item in planning_source if item["system"] == system)
        planning_rows.append(
            {
                "system": system,
                "display_name": row["display_name"],
                "n_runs": int(row["n_runs"]),
                "initial_invalid_n": int(row["initial_invalid_n"]),
                "initial_invalid_rate": as_float(row["initial_invalid_rate"]),
                "repaired_n": int(row["repaired_n"]),
                "repair_rate_among_initial_invalid": as_float(
                    row["repair_rate_among_initial_invalid"]
                ),
                "final_invalid_n": int(row["final_invalid_n"]),
                "final_invalid_rate": as_float(row["final_invalid_rate"]),
            }
        )

    ablation_rows = []
    for row in ablation_source:
        full_rate = next(
            item["rate"]
            for item in primary_rows
            if item["system"] == "B4_FULL_CLOSED_LOOP"
        )
        difference = as_float(row["difference"])
        ablation_rows.append(
            {
                "ablation": row["ablation"],
                "display_name": row["display_name"],
                "effect_definition": row["effect_definition"],
                "estimated_pass_rate": full_rate + difference,
                "difference": difference,
                "ci_lower": as_float(row["ci_low"]),
                "ci_upper": as_float(row["ci_high"]),
                "p_value": as_float(row["p_value"]),
                "p_holm": as_float(row["p_holm"]),
                "n_cases": int(row["n_cases"]),
                "n_bootstrap": int(row["n_bootstrap"]),
                "n_permutation": int(row["n_permutation"]),
            }
        )

    fault_map: dict[str, dict] = {}
    for row in fault_source:
        key = row["fault_type"]
        target = fault_map.setdefault(
            key,
            {
                "fault_type": key,
                "fault_label": row["fault_label"],
                "n_cases": int(row["n_cases"]),
                "scoring_note": row["scoring_note"],
            },
        )
        prefix = (
            "detection"
            if row["metric"] == "failure_detected"
            else "terminal"
        )
        target[f"{prefix}_difference"] = as_float(
            row["difference_b4_minus_b2"]
        )
        target[f"{prefix}_ci_lower"] = as_float(row["ci_low"])
        target[f"{prefix}_ci_upper"] = as_float(row["ci_high"])
    fault_rows = list(fault_map.values())

    write_csv(
        SOURCE_DIR / "system_definitions.csv",
        list(system_rows[0].keys()),
        system_rows,
    )
    write_csv(
        SOURCE_DIR / "primary_endpoint_numeric.csv",
        list(primary_rows[0].keys()),
        primary_rows,
    )
    write_csv(
        SOURCE_DIR / "primary_paired_comparison_numeric.csv",
        list(paired.keys()),
        [paired],
    )
    write_csv(
        SOURCE_DIR / "traceability_reliability_numeric.csv",
        list(trace_rows[0].keys()),
        trace_rows,
    )
    write_csv(
        SOURCE_DIR / "planning_audit_numeric.csv",
        list(planning_rows[0].keys()),
        planning_rows,
    )
    write_csv(
        SOURCE_DIR / "ablation_effects_numeric.csv",
        list(ablation_rows[0].keys()),
        ablation_rows,
    )
    write_csv(
        SOURCE_DIR / "fault_handling_numeric.csv",
        list(fault_rows[0].keys()),
        fault_rows,
    )
    write_csv(
        SOURCE_DIR / "metric_definition_corrections.csv",
        list(corrections_source[0].keys()),
        corrections_source,
    )

    display_primary = []
    for row in primary_rows:
        display_primary.append(
            {
                "System": row["display_name"],
                "Successful runs": f"{row['successes']}/{row['n_runs']}",
                "Composite pass (95% CI)": (
                    f"{row['rate']:.1%} "
                    f"({row['ci_lower']:.1%}–{row['ci_upper']:.1%})"
                ),
                "Cases": row["n_cases"],
                "Repeats per case": row["n_runs"] // row["n_cases"],
            }
        )
    write_csv(
        ROOT / "Table_4_Primary_Agent_Benchmark.csv",
        list(display_primary[0].keys()),
        display_primary,
    )

    display_ablation = []
    for row in ablation_rows:
        display_ablation.append(
            {
                "Ablation": row["display_name"],
                "Estimated composite pass": f"{row['estimated_pass_rate']:.1%}",
                "Change vs full B4 (95% CI), percentage points": (
                    f"{row['difference'] * 100:+.1f} "
                    f"({row['ci_lower'] * 100:+.1f} to "
                    f"{row['ci_upper'] * 100:+.1f})"
                ),
                "Holm-adjusted p": p_display(row["p_holm"]),
            }
        )
    write_csv(
        ROOT / "Table_4_Ablation_Effects.csv",
        list(display_ablation[0].keys()),
        display_ablation,
    )

    display_faults = []
    for row in fault_rows:
        display_faults.append(
            {
                "Fault": row["fault_label"],
                "B4−B2 failure-detection difference (95% CI), pp": (
                    f"{row['detection_difference'] * 100:+.1f} "
                    f"({row['detection_ci_lower'] * 100:+.1f} to "
                    f"{row['detection_ci_upper'] * 100:+.1f})"
                ),
                "B4−B2 correct-terminal difference (95% CI), pp": (
                    f"{row['terminal_difference'] * 100:+.1f} "
                    f"({row['terminal_ci_lower'] * 100:+.1f} to "
                    f"{row['terminal_ci_upper'] * 100:+.1f})"
                ),
                "Cases": row["n_cases"],
                "Scoring note": row["scoring_note"],
            }
        )
    write_csv(
        ROOT / "Table_4_Fault_Handling.csv",
        list(display_faults[0].keys()),
        display_faults,
    )

    provenance_rows = []
    for path, role in [
        (SYSTEM_PATH, "B2 and B4 architecture definitions"),
        (PRIMARY_PATH, "Prespecified composite-pass estimates"),
        (PAIRED_PATH, "Primary paired comparison"),
        (ABLATION_PATH, "Paired component-ablation effects"),
        (TRACEABILITY_PATH, "Comparable traceability and repeatability metrics"),
        (FAULT_PATH, "Frozen fault-injection differences"),
        (PLANNING_PATH, "Post-hoc action-specification audit"),
        (CORRECTIONS_PATH, "Audited metric terminology"),
        (AUDIT_GATE_PATH, "Post-hoc offline audit gate"),
        (FIGURE_QA_PATH, "Figure 4 quality gate"),
        (FORMAL_GATE_PATH, "Complete formal-run gate"),
        (FORMAL_RECORDS_PATH, "Complete frozen 4,860-record JSONL"),
    ]:
        provenance_rows.append(
            {
                "input_file": str(path),
                "role": role,
                "sha256": sha256(path),
            }
        )
    write_csv(
        SOURCE_DIR / "input_provenance.csv",
        ["input_file", "role", "sha256"],
        provenance_rows,
    )

    primary_b2 = next(
        row for row in primary_rows if row["system"] == SYSTEM_ORDER[0]
    )
    primary_b4 = next(
        row for row in primary_rows if row["system"] == SYSTEM_ORDER[1]
    )
    source_gate = {
        "status": "TABLE4_SOURCE_LOCKED",
        "formal_run_status": formal_gate["status"],
        "record_count": formal_gate["record_count"],
        "unique_record_count": formal_gate["unique_record_count"],
        "api_error_count": formal_gate["api_error_count"],
        "figure4_qa_status": figure_qa["status"],
        "offline_audit_status": audit_gate["status"],
        "confirmatory_endpoint_modified": audit_gate[
            "confirmatory_endpoint_modified"
        ],
        "strict_vs_frozen_clean_disagreement_count": audit_gate[
            "strict_vs_frozen_clean_disagreement_count"
        ],
        "systems_exact": [row["system"] for row in system_rows] == SYSTEM_ORDER,
        "primary_paired_runs": paired["n_paired_runs"],
        "primary_b2_rate": primary_b2["rate"],
        "primary_b4_rate": primary_b4["rate"],
        "primary_difference": paired["difference"],
        "primary_p_value": paired["p_value"],
        "ablation_count": len(ablation_rows),
        "fault_type_count": len(fault_rows),
        "fault_metric_count": len(fault_source),
        "schema_metric_in_main_table": False,
        "b2_internal_verifier_encoded_as_zero": False,
        "unsupported_request_scoring_mismatch_preserved": any(
            row["fault_type"] == "UNSUPPORTED_REQUESTED_CLAIM"
            and bool(row["scoring_note"])
            for row in fault_rows
        ),
        "claim_boundary": (
            "Technical agent benchmark only; not clinical utility, diagnostic "
            "accuracy, treatment benefit, deployment readiness, or patient outcomes."
        ),
        "provenance": provenance_rows,
    }
    (SOURCE_DIR / "TABLE4_SOURCE_GATE.json").write_text(
        json.dumps(source_gate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = {
        "table_title": (
            "Table 4. Formal technical benchmark of the verifier-guided "
            "closed-loop Agent system"
        ),
        "subtitle": (
            "Frozen 100-case benchmark with three repeats per case; "
            "complete 4,860-record formal output"
        ),
        "systems": system_rows,
        "primary": primary_rows,
        "paired": paired,
        "traceability": trace_rows,
        "planning": planning_rows,
        "ablations": ablation_rows,
        "faults": fault_rows,
        "corrections": corrections_source,
        "provenance": provenance_rows,
        "source_gate": source_gate,
    }
    (SOURCE_DIR / "table4_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "TABLE4_SOURCE_PREPARED",
                "record_count": source_gate["record_count"],
                "primary_difference": source_gate["primary_difference"],
                "ablations": source_gate["ablation_count"],
                "fault_types": source_gate["fault_type_count"],
                "output_dir": str(ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
