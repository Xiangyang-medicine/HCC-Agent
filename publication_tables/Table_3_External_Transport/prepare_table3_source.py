from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


ROOT = Path(r"F:\ACM\publication_tables\Table_3_External_Transport")
SOURCE_DIR = ROOT / "source_data"
ACM = Path(r"F:\ACM")
FIGURE_ROOT = ACM / "publication_figures" / "Figure_3_External_Transport"

PATIENT_PATH = (
    FIGURE_ROOT / "panel_a_transport_workflow" / "source_data.csv"
)
FLOW_PATH = (
    FIGURE_ROOT
    / "panel_a_transport_workflow"
    / "supplementary_cohort_flow.csv"
)
ASSOCIATION_PATH = FIGURE_ROOT / "panel_b_cohort_flow" / "source_data.csv"
COEFFICIENT_PATH = (
    FIGURE_ROOT / "panel_c_frozen_coefficients" / "source_data.csv"
)
DISCRIMINATION_PATH = (
    FIGURE_ROOT / "panel_d_external_discrimination" / "source_data.csv"
)
MANIFEST_PATH = FIGURE_ROOT / "analysis_manifest.json"
QA_GATE_PATH = FIGURE_ROOT / "FIGURE3_QA_GATE.json"
LEGEND_PATH = FIGURE_ROOT / "FIGURE_3_LEGEND.md"
SOURCE_AUDIT_PATH = (
    ACM
    / "experiments"
    / "phase3b"
    / "microarray_transport"
    / "MICROARRAY_SOURCE_AND_INPUT_AUDIT.json"
)

COHORT_ORDER = [
    ("GSE14520", "GPL3921"),
    ("GSE116174", "GPL570"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frozen_group_result(data: pd.DataFrame) -> dict:
    high = data[data["risk_group"].eq("Higher risk")]
    low = data[data["risk_group"].eq("Lower risk")]
    logrank = logrank_test(
        high["survival_months"],
        low["survival_months"],
        event_observed_A=high["event"],
        event_observed_B=low["event"],
    )
    cox_data = data[["survival_months", "event"]].copy()
    cox_data["higher_risk"] = data["risk_group"].eq("Higher risk").astype(int)
    cox = CoxPHFitter()
    cox.fit(
        cox_data,
        duration_col="survival_months",
        event_col="event",
        formula="higher_risk",
    )
    summary = cox.summary.loc["higher_risk"]
    return {
        "higher_risk_n": int(len(high)),
        "lower_risk_n": int(len(low)),
        "hazard_ratio": float(summary["exp(coef)"]),
        "ci_lower": float(summary["exp(coef) lower 95%"]),
        "ci_upper": float(summary["exp(coef) upper 95%"]),
        "cox_wald_p": float(summary["p"]),
        "logrank_p": float(logrank.p_value),
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    qa_gate = json.loads(QA_GATE_PATH.read_text(encoding="utf-8-sig"))
    source_audit = json.loads(SOURCE_AUDIT_PATH.read_text(encoding="utf-8-sig"))
    if qa_gate.get("success") is not True:
        raise ValueError("Figure 3 QA gate did not pass.")
    if manifest.get("gpl571_performance_displayed") is not False:
        raise ValueError("Canonical Figure 3 unexpectedly displays GPL571.")

    patients = pd.read_csv(PATIENT_PATH)
    associations = pd.read_csv(ASSOCIATION_PATH)
    discrimination = pd.read_csv(DISCRIMINATION_PATH)
    flow = pd.read_csv(FLOW_PATH)
    coefficients = pd.read_csv(COEFFICIENT_PATH)

    cohort_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for cohort, platform in COHORT_ORDER:
        key = patients["cohort"].eq(cohort) & patients["platform"].eq(platform)
        cohort_data = patients.loc[key].copy()
        association = associations[
            associations["cohort"].eq(cohort)
            & associations["platform"].eq(platform)
        ].iloc[0]
        harrell = discrimination[
            discrimination["cohort"].eq(cohort)
            & discrimination["platform"].eq(platform)
            & discrimination["metric"].eq("harrell_c")
        ].iloc[0]
        uno = discrimination[
            discrimination["cohort"].eq(cohort)
            & discrimination["platform"].eq(platform)
            & discrimination["metric"].eq("uno_c")
        ].iloc[0]
        group = frozen_group_result(cohort_data)
        cutoff_values = cohort_data["frozen_tcga_cutoff"].drop_duplicates()
        if len(cutoff_values) != 1:
            raise ValueError(f"{cohort}/{platform} has multiple frozen cutoffs.")

        cohort_rows.append(
            {
                "cohort": cohort,
                "platform": platform,
                "n": int(association["n"]),
                "events": int(association["events"]),
                "event_rate": float(association["event_rate"]),
                "median_observed_time_months": float(
                    association["median_os_months"]
                ),
                "harrell_c": float(harrell["estimate"]),
                "harrell_ci_lower": float(harrell["ci_low"]),
                "harrell_ci_upper": float(harrell["ci_high"]),
                "uno_c": float(uno["estimate"]),
                "uno_ci_lower": float(uno["ci_low"]),
                "uno_ci_upper": float(uno["ci_high"]),
                "uno_tau_months": float(uno["tau_months"]),
                "n_bootstrap": int(uno["n_bootstrap"]),
                "valid_iterations": int(uno["valid_iterations"]),
                "continuous_hr_per_1sd": float(
                    association["hazard_ratio_per_1sd"]
                ),
                "continuous_ci_lower": float(association["ci_low"]),
                "continuous_ci_upper": float(association["ci_high"]),
                "continuous_wald_p": float(association["wald_p"]),
                "continuous_ph_test_p": float(association["ph_test_p"]),
                "risk_standardisation": association["risk_standardisation"],
                "cutpoint_used_for_continuous_effect": bool(
                    association["cutpoint_used"]
                ),
                "external_recalibration": bool(
                    association["external_recalibration"]
                ),
            }
        )
        threshold_rows.append(
            {
                "cohort": cohort,
                "platform": platform,
                **group,
                "frozen_tcga_cutoff": float(cutoff_values.iloc[0]),
                "cutoff_origin": cohort_data["cutoff_origin"].iloc[0],
                "external_outcome_used_for_grouping": bool(
                    cohort_data["external_outcome_used_for_grouping"].iloc[0]
                ),
            }
        )

    cohort_fields = list(cohort_rows[0].keys())
    threshold_fields = list(threshold_rows[0].keys())
    write_csv(
        SOURCE_DIR / "external_cohort_performance_numeric.csv",
        cohort_fields,
        cohort_rows,
    )
    write_csv(
        SOURCE_DIR / "frozen_threshold_results_numeric.csv",
        threshold_fields,
        threshold_rows,
    )

    coefficient_rows = coefficients.to_dict(orient="records")
    write_csv(
        SOURCE_DIR / "frozen_gene_coefficients.csv",
        list(coefficient_rows[0].keys()),
        coefficient_rows,
    )

    display_performance = []
    for row in cohort_rows:
        display_performance.append(
            {
                "External cohort": f"{row['cohort']} ({row['platform']})",
                "Patients": row["n"],
                "Deaths, n (%)": (
                    f"{row['events']} ({row['event_rate']:.1%})"
                ),
                "Harrell C (95% CI)": (
                    f"{row['harrell_c']:.3f} "
                    f"({row['harrell_ci_lower']:.3f}–"
                    f"{row['harrell_ci_upper']:.3f})"
                ),
                "Uno C (95% CI)": (
                    f"{row['uno_c']:.3f} "
                    f"({row['uno_ci_lower']:.3f}–"
                    f"{row['uno_ci_upper']:.3f})"
                ),
                "Uno tau, months": f"{row['uno_tau_months']:.1f}",
                "HR per 1-SD score (95% CI)": (
                    f"{row['continuous_hr_per_1sd']:.2f} "
                    f"({row['continuous_ci_lower']:.2f}–"
                    f"{row['continuous_ci_upper']:.2f})"
                ),
                "Wald p": (
                    "<0.001"
                    if row["continuous_wald_p"] < 0.001
                    else f"{row['continuous_wald_p']:.3f}"
                ),
                "PH-test p": f"{row['continuous_ph_test_p']:.3f}",
            }
        )
    write_csv(
        ROOT / "Table_3_External_Transport_Performance.csv",
        list(display_performance[0].keys()),
        display_performance,
    )

    display_threshold = []
    for row in threshold_rows:
        display_threshold.append(
            {
                "External cohort": f"{row['cohort']} ({row['platform']})",
                "Higher-risk n": row["higher_risk_n"],
                "Lower-risk n": row["lower_risk_n"],
                "Higher vs lower risk HR (95% CI)": (
                    f"{row['hazard_ratio']:.2f} "
                    f"({row['ci_lower']:.2f}–{row['ci_upper']:.2f})"
                ),
                "Log-rank p": (
                    "<0.001"
                    if row["logrank_p"] < 0.001
                    else f"{row['logrank_p']:.3f}"
                ),
                "Frozen cutoff": f"{row['frozen_tcga_cutoff']:.4f}",
                "Cutoff origin": row["cutoff_origin"],
                "External outcome used for grouping": (
                    "No"
                    if not row["external_outcome_used_for_grouping"]
                    else "Yes"
                ),
            }
        )
    write_csv(
        ROOT / "Table_3_Frozen_Threshold_Stratification.csv",
        list(display_threshold[0].keys()),
        display_threshold,
    )

    excluded = {
        "cohort": "GSE14520",
        "platform": "GPL571",
        "complete_os_cases": 21,
        "events": 11,
        "analysis_status": "Not analysed",
        "reason": "Sample size was insufficient for a stable main analysis.",
    }
    write_csv(
        SOURCE_DIR / "excluded_cohort_record.csv",
        list(excluded.keys()),
        [excluded],
    )

    provenance_rows = []
    for path, role in [
        (PATIENT_PATH, "Patient-level frozen-threshold evaluation source"),
        (FLOW_PATH, "External cohort flow"),
        (ASSOCIATION_PATH, "Continuous-score Cox associations"),
        (COEFFICIENT_PATH, "Frozen 15-gene coefficient profile"),
        (DISCRIMINATION_PATH, "Bootstrap discrimination estimates"),
        (MANIFEST_PATH, "Figure 3 locked analysis manifest"),
        (QA_GATE_PATH, "Figure 3 quality gate"),
        (SOURCE_AUDIT_PATH, "External source and input audit"),
        (LEGEND_PATH, "Canonical Figure 3 reporting text"),
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

    nonzero = [
        row["gene"]
        for row in coefficient_rows
        if str(row["nonzero"]).lower() == "true"
    ]
    source_gate = {
        "status": "TABLE3_SOURCE_LOCKED",
        "figure3_qa_status": qa_gate["status"],
        "analysis_scope": manifest["core_claim_scope"],
        "cohorts_exact": [
            f"{row['cohort']}_{row['platform']}" for row in cohort_rows
        ]
        == ["GSE14520_GPL3921", "GSE116174_GPL570"],
        "cohort_count": len(cohort_rows),
        "total_patients": sum(row["n"] for row in cohort_rows),
        "total_events": sum(row["events"] for row in cohort_rows),
        "all_bootstrap_complete": all(
            row["valid_iterations"] == 1000 for row in cohort_rows
        ),
        "single_frozen_cutoff": len(
            {row["frozen_tcga_cutoff"] for row in threshold_rows}
        )
        == 1,
        "cutoff_value": threshold_rows[0]["frozen_tcga_cutoff"],
        "external_outcome_used_for_grouping": any(
            row["external_outcome_used_for_grouping"]
            for row in threshold_rows
        ),
        "external_recalibration": any(
            row["external_recalibration"] for row in cohort_rows
        ),
        "gpl571_performance_included": False,
        "gpl571_exclusion_recorded": True,
        "frozen_nonzero_genes": nonzero,
        "claim_boundary": (
            "External cross-platform evaluation of the frozen gene-only "
            "component; not external validation of M4 or clinical utility."
        ),
        "provenance": provenance_rows,
    }
    (SOURCE_DIR / "TABLE3_SOURCE_GATE.json").write_text(
        json.dumps(source_gate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = {
        "table_title": (
            "Table 3. External cross-platform transport of the frozen "
            "metabolic-gene prognostic component"
        ),
        "subtitle": (
            "Two independent HCC microarray cohorts; outcome-blind scoring "
            "with no external refitting or recalibration"
        ),
        "performance": cohort_rows,
        "threshold": threshold_rows,
        "coefficients": coefficient_rows,
        "excluded": excluded,
        "provenance": provenance_rows,
        "source_gate": source_gate,
        "source_urls": {
            "GSE14520": (
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520"
            ),
            "GSE116174": (
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116174"
            ),
        },
        "audit_protocol": source_audit["protocol"],
    }
    (SOURCE_DIR / "table3_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "TABLE3_SOURCE_PREPARED",
                "cohorts": len(cohort_rows),
                "total_patients": source_gate["total_patients"],
                "total_events": source_gate["total_events"],
                "output_dir": str(ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
