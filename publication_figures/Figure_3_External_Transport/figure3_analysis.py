from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
MODULE_ROOT = PROJECT / "src" / "prognostic_engine" / "src"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

RESULTS = PROJECT / "experiments" / "phase3b" / "microarray_transport"
DERIVATION = PROJECT / "experiments" / "phase3b" / "derivation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(folder: str, rows: list[dict]) -> Path:
    output = ROOT / folder / "source_data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def external_score_association(
    prediction_path: Path,
    cohort: str,
    platform: str,
) -> dict:
    predictions = pd.read_csv(prediction_path)
    risk_sd = float(predictions["risk_score"].std(ddof=1))
    if not np.isfinite(risk_sd) or risk_sd <= 0:
        raise ValueError(f"Invalid risk-score SD for {cohort}/{platform}.")
    analysis = pd.DataFrame(
        {
            "survival_months": predictions["survival_months"].astype(float),
            "event": predictions["event"].astype(int),
            "risk_per_sd": (
                predictions["risk_score"] - predictions["risk_score"].mean()
            )
            / risk_sd,
        }
    )
    cox = CoxPHFitter()
    cox.fit(
        analysis,
        duration_col="survival_months",
        event_col="event",
        formula="risk_per_sd",
    )
    summary = cox.summary.loc["risk_per_sd"]
    ph_test = proportional_hazard_test(
        cox,
        analysis,
        time_transform="rank",
    ).summary.loc["risk_per_sd"]
    return {
        "cohort": cohort,
        "platform": platform,
        "n": len(analysis),
        "events": int(analysis["event"].sum()),
        "event_rate": float(analysis["event"].mean()),
        "median_os_months": float(analysis["survival_months"].median()),
        "hazard_ratio_per_1sd": float(summary["exp(coef)"]),
        "ci_low": float(summary["exp(coef) lower 95%"]),
        "ci_high": float(summary["exp(coef) upper 95%"]),
        "wald_p": float(summary["p"]),
        "ph_test_p": float(ph_test["p"]),
        "risk_standardisation": "within-cohort sample SD",
        "cutpoint_used": False,
        "external_recalibration": False,
    }


def main() -> None:
    manifest_path = DERIVATION / "M2T_CROSSPLATFORM_MANIFEST.json"
    artifact_path = DERIVATION / "m2t_crossplatform_artifact.joblib"
    gate_path = RESULTS / "MICROARRAY_TRANSPORT_GATE.json"
    source_audit_path = RESULTS / "MICROARRAY_SOURCE_AND_INPUT_AUDIT.json"
    derivation_path = (
        PROJECT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if gate.get("success") is not True:
        raise ValueError("The frozen microarray transport gate did not pass.")
    if sha256_file(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError("Frozen M2T artifact hash mismatch.")
    if sha256_file(derivation_path) != manifest["derivation_data_sha256"]:
        raise ValueError("Locked TCGA derivation data hash mismatch.")

    artifact = joblib.load(artifact_path)
    genes = list(manifest["genes"])
    coefficients = np.asarray(artifact.model.coef_, dtype=float).reshape(-1)
    if len(genes) != len(coefficients):
        raise ValueError("Gene and coefficient counts differ.")

    gse14520 = next(
        item
        for item in source_audit["gse14520"]["platforms"]
        if item["platform"] == "GPL3921"
    )
    gse116174 = source_audit["gse116174"]
    flow_rows = [
        {
            "cohort": "GSE14520",
            "platform": "GPL3921",
            "stage_order": 1,
            "stage": "Official arrays",
            "count": gse14520["sample_flow"]["official_expression_arrays"],
            "events": gse14520["sample_flow"]["events"],
            "genes_mapped": 15,
            "genes_required": 15,
        },
        {
            "cohort": "GSE14520",
            "platform": "GPL3921",
            "stage_order": 2,
            "stage": "Clinical match",
            "count": gse14520["sample_flow"]["clinical_matches"],
            "events": gse14520["sample_flow"]["events"],
            "genes_mapped": 15,
            "genes_required": 15,
        },
        {
            "cohort": "GSE14520",
            "platform": "GPL3921",
            "stage_order": 3,
            "stage": "Tumour samples",
            "count": gse14520["sample_flow"]["tumour_matches"],
            "events": gse14520["sample_flow"]["events"],
            "genes_mapped": 15,
            "genes_required": 15,
        },
        {
            "cohort": "GSE14520",
            "platform": "GPL3921",
            "stage_order": 4,
            "stage": "Complete OS",
            "count": gse14520["sample_flow"]["complete_os_cases"],
            "events": gse14520["sample_flow"]["events"],
            "genes_mapped": 15,
            "genes_required": 15,
        },
        {
            "cohort": "GSE116174",
            "platform": "GPL570",
            "stage_order": 1,
            "stage": "Official arrays",
            "count": gse116174["sample_flow"]["official_expression_arrays"],
            "events": gse116174["sample_flow"]["events"],
            "genes_mapped": 15,
            "genes_required": 15,
        },
        {
            "cohort": "GSE116174",
            "platform": "GPL570",
            "stage_order": 2,
            "stage": "Clinical match",
            "count": gse116174["sample_flow"][
                "unique_expression_to_clinical_matches"
            ],
            "events": gse116174["sample_flow"]["events"],
            "genes_mapped": 15,
            "genes_required": 15,
        },
        {
            "cohort": "GSE116174",
            "platform": "GPL570",
            "stage_order": 3,
            "stage": "Complete OS",
            "count": gse116174["sample_flow"]["complete_os_cases"],
            "events": gse116174["sample_flow"]["events"],
            "genes_mapped": 15,
            "genes_required": 15,
        },
    ]
    flow_path = (
        ROOT
        / "panel_a_transport_workflow"
        / "supplementary_cohort_flow.csv"
    )
    pd.DataFrame(flow_rows).to_csv(flow_path, index=False)

    prediction_specs = [
        ("GSE14520", "GPL3921", "GSE14520_GPL3921"),
        ("GSE116174", "GPL570", "GSE116174_GPL570"),
    ]
    prediction_paths = [
        RESULTS / f"{label}_m2t_evaluation_predictions.csv"
        for _, _, label in prediction_specs
    ]
    association_rows = [
        external_score_association(path, cohort, platform)
        for (cohort, platform, _), path in zip(
            prediction_specs, prediction_paths
        )
    ]
    association_path = write_csv("panel_b_cohort_flow", association_rows)

    derivation = pd.read_parquet(derivation_path)
    gene_columns = [f"{gene}_log2tpm" for gene in genes]
    derivation_z = (
        derivation[gene_columns] - derivation[gene_columns].mean()
    ) / derivation[gene_columns].std(ddof=1)
    derivation_risk = artifact.model.predict(derivation_z.to_numpy())
    frozen_cutoff = float(np.median(derivation_risk))
    stratification_frames = []
    for (cohort, platform, _), prediction_path in zip(
        prediction_specs, prediction_paths
    ):
        predictions = pd.read_csv(prediction_path)
        predictions = predictions[
            [
                "case_id",
                "source_sample_id",
                "survival_months",
                "event",
                "risk_score",
            ]
        ].copy()
        predictions.insert(1, "cohort", cohort)
        predictions.insert(2, "platform", platform)
        predictions["frozen_tcga_cutoff"] = frozen_cutoff
        predictions["risk_group"] = np.where(
            predictions["risk_score"] > frozen_cutoff,
            "Higher risk",
            "Lower risk",
        )
        predictions["cutoff_origin"] = "TCGA derivation median"
        predictions["external_outcome_used_for_grouping"] = False
        stratification_frames.append(predictions)
    stratification_path = (
        ROOT / "panel_a_transport_workflow" / "source_data.csv"
    )
    pd.concat(stratification_frames, ignore_index=True).to_csv(
        stratification_path, index=False
    )

    coefficient_rows = []
    for order, (gene, coefficient) in enumerate(
        zip(genes, coefficients), start=1
    ):
        coefficient_rows.append(
            {
                "feature_order": order,
                "gene": gene,
                "coefficient": float(coefficient),
                "nonzero": bool(not np.isclose(coefficient, 0.0)),
                "direction": (
                    "higher_risk"
                    if coefficient > 0
                    else "lower_risk"
                    if coefficient < 0
                    else "zero"
                ),
            }
        )
    coefficient_path = write_csv(
        "panel_c_frozen_coefficients", coefficient_rows
    )

    cohort_specs = [
        ("GSE14520", "GPL3921", 221, 85, "GSE14520_GPL3921"),
        ("GSE116174", "GPL570", 64, 27, "GSE116174_GPL570"),
    ]
    metric_rows: list[dict] = []
    input_paths = [
        manifest_path,
        artifact_path,
        gate_path,
        source_audit_path,
        derivation_path,
        *prediction_paths,
    ]
    for cohort, platform, expected_n, expected_events, label in cohort_specs:
        result_path = RESULTS / f"{label}_M2T_EVALUATION.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        input_paths.append(result_path)
        if payload["n"] != expected_n or payload["events"] != expected_events:
            raise ValueError(f"Unexpected cohort counts for {label}.")
        for metric, display in [
            ("harrell_c", "Harrell C"),
            ("uno_c", "Uno C"),
        ]:
            values = payload["bootstrap"]["metrics"][metric]
            metric_rows.append(
                {
                    "cohort": cohort,
                    "platform": platform,
                    "n": payload["n"],
                    "events": payload["events"],
                    "metric": metric,
                    "metric_display": display,
                    "estimate": values["point_estimate"],
                    "ci_low": values["ci95"][0],
                    "ci_high": values["ci95"][1],
                    "n_bootstrap": payload["bootstrap"]["n_bootstrap"],
                    "valid_iterations": payload["bootstrap"][
                        "valid_iterations"
                    ],
                    "tau_months": (
                        payload["bootstrap"]["tau_months"]
                        if metric == "uno_c"
                        else np.nan
                    ),
                    "analysis_scope": (
                        "secondary_exploratory_cross_platform"
                    ),
                }
            )
    metric_path = write_csv(
        "panel_d_external_discrimination", metric_rows
    )

    outputs = [
        stratification_path,
        flow_path,
        association_path,
        coefficient_path,
        metric_path,
    ]
    analysis_manifest = {
        "status": "FIGURE3_SOURCE_DATA_GENERATED",
        "core_claim_scope": (
            "secondary_exploratory_cross_platform_transport"
        ),
        "external_cohorts_in_performance_panel": 2,
        "pooled_patient_analysis": False,
        "external_outcomes_used_for_model_fitting": False,
        "external_cutpoint_selected": False,
        "risk_group_cutpoint_origin": "TCGA derivation median",
        "risk_group_cutpoint": frozen_cutoff,
        "external_outcome_used_for_risk_grouping": False,
        "external_evaluation_cox_association_reported": True,
        "external_evaluation_cox_recalibrated_model": False,
        "gpl571_performance_displayed": False,
        "input_hashes": {
            str(path.relative_to(PROJECT)): sha256_file(path)
            for path in input_paths
        },
        "output_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in outputs
        },
    }
    (ROOT / "analysis_manifest.json").write_text(
        json.dumps(
            analysis_manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
