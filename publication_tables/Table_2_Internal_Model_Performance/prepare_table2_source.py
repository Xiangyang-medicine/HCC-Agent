from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"F:\ACM\publication_tables\Table_2_Internal_Model_Performance")
SOURCE_DIR = ROOT / "source_data"
ACM = Path(r"F:\ACM")

METRICS_PATH = ACM / "experiments" / "phase3a" / "formal" / "metrics_summary.json"
COMPARISONS_PATH = (
    ACM / "experiments" / "phase3a" / "formal" / "model_comparisons_v6.csv"
)
AUDIT_PATH = ACM / "experiments" / "phase3a" / "formal" / "AUDIT_REPORT_V5.json"
SENSITIVITY_PATH = (
    ACM
    / "experiments"
    / "phase3a"
    / "sensitivity"
    / "SENSITIVITY_SUMMARY_V2.csv"
)
FIGURE_MANIFEST_PATH = (
    ACM
    / "publication_figures"
    / "Figure_2_TCGA_Internal_Validation"
    / "analysis_manifest.json"
)

MODEL_META = {
    "M1_clinical_cox": {
        "model_id": "M1",
        "display_name": "Clinical Cox PH",
        "predictors": "Age, AJCC stage, tumour grade",
        "algorithm": "Cox proportional hazards",
        "role": "Clinical reference",
    },
    "M2_gene_elasticnet": {
        "model_id": "M2",
        "display_name": "Gene elastic-net",
        "predictors": "15 locked metabolic genes",
        "algorithm": "Elastic-net Cox",
        "role": "Gene-only comparator",
    },
    "M3_combined_elasticnet": {
        "model_id": "M3",
        "display_name": "Combined elastic-net",
        "predictors": "Clinical variables + 15 genes",
        "algorithm": "Elastic-net Cox",
        "role": "Combined linear comparator",
    },
    "M4_combined_rsf": {
        "model_id": "M4",
        "display_name": "Combined RSF",
        "predictors": "Clinical variables + 15 genes",
        "algorithm": "Random survival forest",
        "role": "Provisional primary candidate",
    },
    "M5_deepsurv": {
        "model_id": "M5",
        "display_name": "DeepSurv",
        "predictors": "Clinical variables + 15 genes",
        "algorithm": "Neural Cox model",
        "role": "Exploratory deep comparator",
    },
}

METRIC_FIELDS = [
    "harrell_c",
    "uno_c",
    "auc_12m",
    "auc_36m",
    "auc_60m",
    "brier_12m",
    "brier_36m",
    "brier_60m",
    "ibs",
]

METRIC_LABELS = {
    "harrell_c": "Harrell C",
    "uno_c": "Uno C",
    "auc_12m": "AUC, 12 months",
    "auc_36m": "AUC, 36 months",
    "auc_60m": "AUC, 60 months",
    "brier_12m": "Brier score, 12 months",
    "brier_36m": "Brier score, 36 months",
    "brier_60m": "Brier score, 60 months",
    "ibs": "Integrated Brier score",
}


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


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_document = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    performance_rows: list[dict] = []
    for model_key, meta in MODEL_META.items():
        model_metrics = metrics_document["metrics"][model_key]
        row = {
            "model_key": model_key,
            **meta,
            "n_outer_folds": model_metrics["harrell_c"]["n_folds"],
        }
        for metric in METRIC_FIELDS:
            row[f"{metric}_mean"] = model_metrics[metric]["mean"]
            row[f"{metric}_sd"] = model_metrics[metric]["std"]
            row[f"{metric}_min"] = model_metrics[metric]["min"]
            row[f"{metric}_max"] = model_metrics[metric]["max"]
        performance_rows.append(row)

    with COMPARISONS_PATH.open(encoding="utf-8-sig", newline="") as handle:
        source_comparisons = list(csv.DictReader(handle))

    formal_comparisons: list[dict] = []
    for row in source_comparisons:
        if row["type"] != "Formal":
            continue
        model_a = MODEL_META[row["model_a"]]["model_id"]
        model_b = MODEL_META[row["model_b"]]["model_id"]
        formal_comparisons.append(
            {
                "comparison": f"{model_a} vs {model_b}",
                "metric": METRIC_LABELS[row["metric"]],
                "metric_key": row["metric"],
                "model_a": model_a,
                "model_b": model_b,
                "mean_difference": float(row["mean_diff"]),
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
                "p_value_raw": float(row["p_value_raw"]),
                "p_value_adjusted": float(row["p_value_adjusted"]),
                "significant_adjusted": row["significant_adjusted"].lower() == "true",
                "iterations_valid": int(row["iterations_valid"]),
                "n_patients": int(row["n_patients"]),
                "n_repeats": int(row["n_repeats"]),
                "n_folds": int(row["n_folds"]),
                "ipcw_source": row["ipcw_source"],
            }
        )

    performance_fields = [
        "model_key",
        "model_id",
        "display_name",
        "predictors",
        "algorithm",
        "role",
        "n_outer_folds",
    ]
    for metric in METRIC_FIELDS:
        performance_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_sd",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    write_csv(
        SOURCE_DIR / "model_performance_numeric.csv",
        performance_fields,
        performance_rows,
    )

    comparison_fields = [
        "comparison",
        "metric",
        "metric_key",
        "model_a",
        "model_b",
        "mean_difference",
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
    write_csv(
        SOURCE_DIR / "formal_paired_comparisons.csv",
        comparison_fields,
        formal_comparisons,
    )

    display_performance = []
    for row in performance_rows:
        display_performance.append(
            {
                "Model": f"{row['model_id']} {row['display_name']}",
                "Predictors": row["predictors"],
                "Algorithm": row["algorithm"],
                "Harrell C, mean (SD)": (
                    f"{row['harrell_c_mean']:.3f} ({row['harrell_c_sd']:.3f})"
                ),
                "Uno C, mean (SD)": (
                    f"{row['uno_c_mean']:.3f} ({row['uno_c_sd']:.3f})"
                ),
                "AUC 12 months, mean (SD)": (
                    f"{row['auc_12m_mean']:.3f} ({row['auc_12m_sd']:.3f})"
                ),
                "AUC 36 months, mean (SD)": (
                    f"{row['auc_36m_mean']:.3f} ({row['auc_36m_sd']:.3f})"
                ),
                "AUC 60 months, mean (SD)": (
                    f"{row['auc_60m_mean']:.3f} ({row['auc_60m_sd']:.3f})"
                ),
                "IBS, mean (SD)": f"{row['ibs_mean']:.3f} ({row['ibs_sd']:.3f})",
                "Model role": row["role"],
            }
        )
    write_csv(
        ROOT / "Table_2_Internal_Model_Performance.csv",
        list(display_performance[0].keys()),
        display_performance,
    )

    display_comparisons = []
    for row in formal_comparisons:
        display_comparisons.append(
            {
                "Comparison": row["comparison"],
                "Metric": row["metric"],
                "Mean difference (95% CI)": (
                    f"{row['mean_difference']:+.3f} "
                    f"({row['ci_lower']:+.3f} to {row['ci_upper']:+.3f})"
                ),
                "Raw p": f"{row['p_value_raw']:.4f}",
                "Bonferroni-adjusted p": f"{row['p_value_adjusted']:.4f}",
                "Significant after adjustment": (
                    "Yes" if row["significant_adjusted"] else "No"
                ),
            }
        )
    write_csv(
        ROOT / "Table_2_Model_Comparisons.csv",
        list(display_comparisons[0].keys()),
        display_comparisons,
    )

    provenance_rows = []
    for path, role in [
        (METRICS_PATH, "Canonical 25-fold model metrics"),
        (COMPARISONS_PATH, "Canonical v6 paired-bootstrap comparisons"),
        (AUDIT_PATH, "Phase 3A method-closure audit"),
        (SENSITIVITY_PATH, "Sensitivity-analysis summary"),
        (FIGURE_MANIFEST_PATH, "Figure 2 analysis manifest"),
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

    gate = {
        "status": "TABLE2_SOURCE_LOCKED",
        "phase3a_status": audit["status"],
        "models_exact": [row["model_id"] for row in performance_rows]
        == ["M1", "M2", "M3", "M4", "M5"],
        "all_models_have_25_folds": all(
            row["n_outer_folds"] == 25 for row in performance_rows
        ),
        "formal_comparison_count": len(formal_comparisons),
        "all_comparisons_have_1000_resamples": all(
            row["iterations_valid"] == 1000 for row in formal_comparisons
        ),
        "all_comparisons_use_363_patients": all(
            row["n_patients"] == 363 for row in formal_comparisons
        ),
        "canonical_comparison_version": "v6",
        "m4_vs_m1_adjusted_significance": {
            row["metric_key"]: row["significant_adjusted"]
            for row in formal_comparisons
            if row["comparison"] == "M4 vs M1"
        },
        "m5_vs_m1_adjusted_significance": {
            row["metric_key"]: row["significant_adjusted"]
            for row in formal_comparisons
            if row["comparison"] == "M5 vs M1"
        },
        "provenance": provenance_rows,
    }
    (SOURCE_DIR / "TABLE2_SOURCE_GATE.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = {
        "table_title": (
            "Table 2. Internal performance and prespecified paired model "
            "comparisons in TCGA-LIHC"
        ),
        "subtitle": (
            "Five repeated 5-fold nested cross-validation "
            "(25 outer test folds; N=363)"
        ),
        "performance": performance_rows,
        "comparisons": formal_comparisons,
        "provenance": provenance_rows,
        "source_gate": gate,
    }
    (SOURCE_DIR / "table2_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "TABLE2_SOURCE_PREPARED",
                "models": len(performance_rows),
                "formal_comparisons": len(formal_comparisons),
                "output_dir": str(ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
