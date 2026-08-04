from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
FORMAL = Path(r"F:\ACM\experiments\phase3a\formal")
SENSITIVITY = Path(r"F:\ACM\experiments\phase3a\sensitivity")

METRICS_PATH = FORMAL / "metrics_summary.json"
OOF_PATH = FORMAL / "oof_predictions.csv"
COMPARISON_PATH = FORMAL / "model_comparisons_v6.csv"
SENSITIVITY_PATH = SENSITIVITY / "SENSITIVITY_SUMMARY_V2.csv"
AUDIT_PATH = FORMAL / "AUDIT_REPORT_V5.json"

MODEL_ORDER = [
    "M1_clinical_cox",
    "M2_gene_elasticnet",
    "M3_combined_elasticnet",
    "M4_combined_rsf",
    "M5_deepsurv",
]
SHORT = {
    "M1_clinical_cox": "M1",
    "M2_gene_elasticnet": "M2",
    "M3_combined_elasticnet": "M3",
    "M4_combined_rsf": "M4",
    "M5_deepsurv": "M5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def panel_a(metrics: dict) -> Path:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        for metric_key, metric_label in [
            ("harrell_c", "Harrell C"),
            ("uno_c", "Uno C"),
        ]:
            values = metrics["metrics"][model][metric_key]["per_fold"]
            if len(values) != 25:
                raise ValueError(f"{model} {metric_key}: expected 25 folds, got {len(values)}")
            for index, value in enumerate(values):
                rows.append(
                    {
                        "model": model,
                        "model_short": SHORT[model],
                        "metric": metric_label,
                        "repeat": index // 5 + 1,
                        "fold": index % 5 + 1,
                        "outer_fold_index": index + 1,
                        "value": float(value),
                    }
                )
    output = ROOT / "panel_a_model_discrimination" / "source_data.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def panel_b() -> Path:
    comparisons = pd.read_csv(COMPARISON_PATH)
    formal = comparisons.loc[comparisons["type"].eq("Formal")].copy()
    formal["metric_label"] = formal["metric"].map(
        {"harrell_c": "Harrell C", "uno_c": "Uno C"}
    )
    formal["model_a_short"] = formal["model_a"].map(SHORT)
    formal["model_b_short"] = formal["model_b"].map(SHORT)
    formal["comparison_label"] = (
        formal["model_a_short"] + " vs " + formal["model_b_short"]
    )
    keep = [
        "comparison_label",
        "metric_label",
        "mean_diff",
        "ci_lower",
        "ci_upper",
        "p_value_raw",
        "p_value_adjusted",
        "significant_adjusted",
        "iterations_valid",
        "n_patients",
        "n_repeats",
        "n_folds",
        "ipcw_source",
    ]
    output = ROOT / "panel_b_paired_differences" / "source_data.csv"
    formal[keep].to_csv(output, index=False)
    return output


def panel_c(oof: pd.DataFrame) -> list[Path]:
    m4 = oof.loc[oof["model"].eq("M4_combined_rsf")].copy()
    if len(m4) != 1815:
        raise ValueError(f"M4 must contain 1815 OOF predictions, got {len(m4)}")
    if m4[["risk_score", "survival_months", "event"]].isna().any().any():
        raise ValueError("Missing M4 OOF risk or outcome values")

    # Risk scales differ across independently fitted outer-fold models. Rank each
    # test fold without using outcomes, then average the five OOF ranks per patient.
    m4["fold_risk_percentile"] = m4.groupby(["repeat", "fold"])["risk_score"].transform(
        lambda values: pd.Series(
            rankdata(values, method="average"), index=values.index
        )
        / (len(values) + 1.0)
    )
    patients = (
        m4.groupby("case_id", as_index=False)
        .agg(
            survival_months=("survival_months", "first"),
            event=("event", "first"),
            oof_risk_percentile=("fold_risk_percentile", "mean"),
            oof_repeats=("repeat", "nunique"),
        )
        .sort_values("case_id")
    )
    if len(patients) != 363 or not patients["oof_repeats"].eq(5).all():
        raise ValueError("Expected 363 patients with five OOF predictions each")

    cutoff = float(patients["oof_risk_percentile"].median())
    patients["risk_group"] = np.where(
        patients["oof_risk_percentile"] > cutoff, "High", "Low"
    )
    patient_path = ROOT / "panel_c_oof_survival" / "source_data_patients.csv"
    patients.to_csv(patient_path, index=False)

    curves: list[pd.DataFrame] = []
    at_risk_rows: list[dict] = []
    time_grid = [0, 24, 48, 72, 96]
    for group in ["Low", "High"]:
        subset = patients.loc[patients["risk_group"].eq(group)]
        km = KaplanMeierFitter(alpha=0.05)
        km.fit(
            subset["survival_months"],
            event_observed=subset["event"],
            label=group,
        )
        sf = km.survival_function_.reset_index()
        ci = km.confidence_interval_.reset_index()
        curve = pd.DataFrame(
            {
                "timeline_months": sf["timeline"],
                "survival_probability": sf[group],
                "ci_lower": ci.iloc[:, 1],
                "ci_upper": ci.iloc[:, 2],
                "risk_group": group,
            }
        )
        curves.append(curve)
        for time in time_grid:
            at_risk_rows.append(
                {
                    "risk_group": group,
                    "time_months": time,
                    "n_at_risk": int((subset["survival_months"] >= time).sum()),
                }
            )

    curve_path = ROOT / "panel_c_oof_survival" / "source_data_km_curves.csv"
    pd.concat(curves, ignore_index=True).to_csv(curve_path, index=False)
    at_risk_path = ROOT / "panel_c_oof_survival" / "source_data_at_risk.csv"
    pd.DataFrame(at_risk_rows).to_csv(at_risk_path, index=False)

    high = patients.loc[patients["risk_group"].eq("High")]
    low = patients.loc[patients["risk_group"].eq("Low")]
    logrank = logrank_test(
        high["survival_months"],
        low["survival_months"],
        event_observed_A=high["event"],
        event_observed_B=low["event"],
    )
    cox_data = patients[["survival_months", "event"]].copy()
    cox_data["high_risk"] = patients["risk_group"].eq("High").astype(int)
    cox = CoxPHFitter().fit(
        cox_data, duration_col="survival_months", event_col="event"
    )
    row = cox.summary.loc["high_risk"]
    stats_path = ROOT / "panel_c_oof_survival" / "statistics.json"
    write_json(
        stats_path,
        {
            "n_patients": int(len(patients)),
            "n_high": int(len(high)),
            "n_low": int(len(low)),
            "events_high": int(high["event"].sum()),
            "events_low": int(low["event"].sum()),
            "outcome_blind_cutoff": "median of patient-level mean OOF fold-risk percentiles",
            "cutoff_value": cutoff,
            "cox_hr_high_vs_low": float(row["exp(coef)"]),
            "cox_hr_ci_lower": float(row["exp(coef) lower 95%"]),
            "cox_hr_ci_upper": float(row["exp(coef) upper 95%"]),
            "cox_p_value": float(row["p"]),
            "logrank_chi_square": float(logrank.test_statistic),
            "logrank_p_value": float(logrank.p_value),
            "interpretation_boundary": (
                "Internal OOF association only; the median grouping rule is not "
                "claimed as an externally validated clinical cutoff."
            ),
        },
    )
    return [patient_path, curve_path, at_risk_path, stats_path]


def panel_d(metrics: dict) -> Path:
    rows: list[dict] = []
    for model in ["M1_clinical_cox", "M4_combined_rsf"]:
        for horizon, metric_key in [
            (12, "brier_12m"),
            (36, "brier_36m"),
            (60, "brier_60m"),
        ]:
            values = metrics["metrics"][model][metric_key]["per_fold"]
            for index, value in enumerate(values):
                rows.append(
                    {
                        "model": model,
                        "model_short": SHORT[model],
                        "horizon_months": horizon,
                        "repeat": index // 5 + 1,
                        "fold": index % 5 + 1,
                        "outer_fold_index": index + 1,
                        "brier_score": float(value),
                    }
                )
    output = ROOT / "panel_d_prediction_error" / "source_data.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    ibs = {
        model: metrics["metrics"][model]["ibs"]["mean"]
        for model in ["M1_clinical_cox", "M4_combined_rsf"]
    }
    write_json(
        ROOT / "panel_d_prediction_error" / "ibs_summary.json",
        {
            "M1_clinical_cox": float(ibs["M1_clinical_cox"]),
            "M4_combined_rsf": float(ibs["M4_combined_rsf"]),
            "direction": "lower_is_better",
            "definition": "mean integrated Brier score across 25 outer folds",
        },
    )
    return output


def panel_e() -> Path:
    source = pd.read_csv(SENSITIVITY_PATH)
    model_map = {
        "M1 (Clinical Cox)": "M1",
        "M2 (Gene Elasticnet)": "M2",
        "M3 (Combined Elasticnet)": "M3",
        "M4 (Combined RSF)": "M4",
        "M5 (DeepSurv)": "M5",
    }
    rows: list[dict] = []
    for _, record in source.iterrows():
        for analysis in ["SA1", "SA2", "SA3"]:
            for metric, value_suffix, rank_suffix, direction in [
                ("Harrell C", "Harrell_C", "Harrell_Rank", "higher_is_better"),
                ("Uno C", "Uno_C", "Uno_Rank", "higher_is_better"),
                ("IBS", "IBS", "IBS_Rank", "lower_is_better"),
            ]:
                rows.append(
                    {
                        "model": record["Model"],
                        "model_short": model_map[record["Model"]],
                        "analysis": analysis,
                        "analysis_definition": {
                            "SA1": "primary cohort (N=363)",
                            "SA2": "age <18 excluded (N=361)",
                            "SA3": "complete stage/grade cases (N=338)",
                        }[analysis],
                        "metric": metric,
                        "value": float(record[f"{analysis}_{value_suffix}"]),
                        "rank": int(record[f"{analysis}_{rank_suffix}"]),
                        "direction": direction,
                    }
                )
    output = ROOT / "panel_e_sensitivity" / "source_data.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    for path in [METRICS_PATH, OOF_PATH, COMPARISON_PATH, SENSITIVITY_PATH, AUDIT_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    with METRICS_PATH.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    oof = pd.read_csv(OOF_PATH)

    outputs = [
        panel_a(metrics),
        panel_b(),
        *panel_c(oof),
        panel_d(metrics),
        panel_e(),
    ]
    manifest = {
        "figure": "Figure 2 — TCGA-LIHC internal validation",
        "status": "SOURCE_DATA_GENERATED_FROM_CANONICAL_LOCKED_OUTPUTS",
        "input_files": {
            str(path): sha256(path)
            for path in [
                METRICS_PATH,
                OOF_PATH,
                COMPARISON_PATH,
                SENSITIVITY_PATH,
                AUDIT_PATH,
            ]
        },
        "output_files": {str(path): sha256(path) for path in outputs},
        "critical_constraints": [
            "model_comparisons_v6 is canonical; v5 and earlier are not used",
            "all model summaries use 25 outer folds from 5x5 repeated nested CV",
            "OOF risk grouping uses outcome-blind within-fold ranks",
            "no result is simulated or manually entered",
            "M4 is described as provisional primary candidate, not statistically superior to M1 after correction",
        ],
    }
    write_json(ROOT / "analysis_manifest.json", manifest)


if __name__ == "__main__":
    main()
