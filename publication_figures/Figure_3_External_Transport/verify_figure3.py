from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        ROOT / "Figure_3_External_Transport.svg",
        ROOT / "Figure_3_External_Transport.pdf",
        ROOT / "Figure_3_External_Transport.png",
        ROOT / "Figure_3_External_Transport.tiff",
        ROOT / "analysis_manifest.json",
        ROOT / "panel_a_transport_workflow" / "source_data.csv",
        ROOT
        / "panel_a_transport_workflow"
        / "supplementary_cohort_flow.csv",
        ROOT / "panel_b_cohort_flow" / "source_data.csv",
        ROOT / "panel_c_frozen_coefficients" / "source_data.csv",
        ROOT / "panel_d_external_discrimination" / "source_data.csv",
        ROOT / "FIGURE_3_LEGEND.md",
        ROOT / "FIGURE_CONTRACT.md",
    ]
    checks: dict[str, bool] = {}
    for path in required:
        checks[f"exists_{path.name}"] = path.exists() and path.stat().st_size > 0

    manifest = json.loads(
        (ROOT / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    checks["analysis_status"] = (
        manifest.get("status") == "FIGURE3_SOURCE_DATA_GENERATED"
    )
    checks["two_external_performance_cohorts"] = (
        manifest.get("external_cohorts_in_performance_panel") == 2
    )
    checks["no_pooled_patient_analysis"] = (
        manifest.get("pooled_patient_analysis") is False
    )
    checks["outcome_blind_model_fitting"] = (
        manifest.get("external_outcomes_used_for_model_fitting") is False
    )
    checks["no_external_cutpoint"] = (
        manifest.get("external_cutpoint_selected") is False
    )
    checks["gpl571_performance_not_displayed"] = (
        manifest.get("gpl571_performance_displayed") is False
    )

    stratification = pd.read_csv(
        ROOT / "panel_a_transport_workflow" / "source_data.csv"
    )
    checks["stratification_rows_285"] = len(stratification) == 285
    checks["stratification_cohorts_exact"] = set(
        zip(stratification["cohort"], stratification["platform"])
    ) == {
        ("GSE14520", "GPL3921"),
        ("GSE116174", "GPL570"),
    }
    checks["stratification_unique_cases"] = not stratification.duplicated(
        ["cohort", "platform", "case_id"]
    ).any()
    checks["frozen_cutoff_single_value"] = (
        stratification["frozen_tcga_cutoff"].nunique() == 1
        and np.isclose(
            stratification["frozen_tcga_cutoff"].iloc[0],
            manifest["risk_group_cutpoint"],
        )
    )
    checks["cutoff_origin_tcga_only"] = (
        stratification["cutoff_origin"].eq(
            "TCGA derivation median"
        ).all()
        and manifest.get("risk_group_cutpoint_origin")
        == "TCGA derivation median"
    )
    checks["external_outcome_not_used_for_grouping"] = bool(
        ~stratification[
            "external_outcome_used_for_grouping"
        ].astype(bool).any()
        and manifest.get("external_outcome_used_for_risk_grouping") is False
    )
    checks["risk_groups_exact"] = set(stratification["risk_group"]) == {
        "Higher risk",
        "Lower risk",
    }
    group_counts = stratification.groupby(
        ["cohort", "risk_group"]
    ).size()
    checks["risk_group_counts_expected"] = (
        group_counts[("GSE14520", "Higher risk")] == 113
        and group_counts[("GSE14520", "Lower risk")] == 108
        and group_counts[("GSE116174", "Higher risk")] == 35
        and group_counts[("GSE116174", "Lower risk")] == 29
    )
    checks["stratification_values_finite"] = bool(
        np.isfinite(
            stratification[
                [
                    "survival_months",
                    "event",
                    "risk_score",
                    "frozen_tcga_cutoff",
                ]
            ].to_numpy()
        ).all()
    )

    flow = pd.read_csv(
        ROOT
        / "panel_a_transport_workflow"
        / "supplementary_cohort_flow.csv"
    )
    checks["sample_flow_rows_seven"] = len(flow) == 7
    checks["sample_flow_cohorts_exact"] = set(flow["cohort"]) == {
        "GSE14520",
        "GSE116174",
    }
    checks["retained_external_counts"] = (
        (
            (flow["cohort"].eq("GSE14520"))
            & (flow["platform"].eq("GPL3921"))
            & (flow["stage"].eq("Complete OS"))
            & (flow["count"].eq(221))
            & (flow["events"].eq(85))
        ).any()
        and (
            (flow["cohort"].eq("GSE116174"))
            & (flow["platform"].eq("GPL570"))
            & (flow["stage"].eq("Complete OS"))
            & (flow["count"].eq(64))
            & (flow["events"].eq(27))
        ).any()
    )
    checks["complete_gene_coverage"] = bool(
        flow["genes_mapped"].eq(15).all()
        and flow["genes_required"].eq(15).all()
    )

    associations = pd.read_csv(
        ROOT / "panel_b_cohort_flow" / "source_data.csv"
    )
    checks["two_continuous_score_associations"] = len(associations) == 2
    checks["association_cohorts_exact"] = set(associations["cohort"]) == {
        "GSE14520",
        "GSE116174",
    }
    checks["association_values_finite"] = bool(
        np.isfinite(
            associations[
                [
                    "hazard_ratio_per_1sd",
                    "ci_low",
                    "ci_high",
                    "wald_p",
                    "ph_test_p",
                ]
            ].to_numpy()
        ).all()
    )
    checks["association_ci_contains_estimate"] = bool(
        (
            (associations["ci_low"] <= associations["hazard_ratio_per_1sd"])
            & (
                associations["hazard_ratio_per_1sd"]
                <= associations["ci_high"]
            )
        ).all()
    )
    checks["association_is_cutpoint_free"] = bool(
        ~associations["cutpoint_used"].astype(bool).any()
    )
    checks["association_is_not_recalibration"] = bool(
        ~associations["external_recalibration"].astype(bool).any()
    )

    coefficients = pd.read_csv(
        ROOT / "panel_c_frozen_coefficients" / "source_data.csv"
    )
    checks["fifteen_coefficients"] = len(coefficients) == 15
    nonzero = coefficients.loc[coefficients["nonzero"], "gene"].tolist()
    checks["two_nonzero_coefficients"] = set(nonzero) == {"PKM", "LDHA"}
    checks["finite_coefficients"] = bool(
        np.isfinite(coefficients["coefficient"].to_numpy()).all()
    )

    metrics = pd.read_csv(
        ROOT / "panel_d_external_discrimination" / "source_data.csv"
    )
    checks["four_external_metrics"] = len(metrics) == 4
    checks["performance_cohorts_exact"] = set(metrics["cohort"]) == {
        "GSE14520",
        "GSE116174",
    }
    checks["no_gpl571_metric"] = not metrics["platform"].eq("GPL571").any()
    checks["bootstrap_1000_valid"] = bool(
        metrics["n_bootstrap"].eq(1000).all()
        and metrics["valid_iterations"].eq(1000).all()
    )
    checks["finite_metric_values"] = bool(
        np.isfinite(
            metrics[["estimate", "ci_low", "ci_high"]].to_numpy()
        ).all()
    )
    checks["ci_contains_estimate"] = bool(
        (
            (metrics["ci_low"] <= metrics["estimate"])
            & (metrics["estimate"] <= metrics["ci_high"])
        ).all()
    )

    svg_text = (ROOT / "Figure_3_External_Transport.svg").read_text(
        encoding="utf-8"
    )
    checks["svg_editable_text"] = "<text" in svg_text
    checks["svg_has_four_panel_labels"] = all(
        re.search(rf">\s*{letter}\s*</text>", svg_text)
        for letter in "abcd"
    )

    with Image.open(ROOT / "Figure_3_External_Transport.tiff") as image:
        checks["tiff_minimum_dimensions"] = (
            image.width >= 4000 and image.height >= 2800
        )
        dpi = image.info.get("dpi", (0, 0))
        checks["tiff_600dpi"] = round(float(dpi[0])) == 600
    with Image.open(ROOT / "Figure_3_External_Transport.png") as image:
        dpi = image.info.get("dpi", (0, 0))
        checks["png_300dpi"] = round(float(dpi[0])) == 300

    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    gate = {
        "status": "FIGURE3_QA_PASSED" if not failed else "FIGURE3_QA_FAILED",
        "success": not failed,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in required[:4]
            if path.exists()
        },
    }
    (ROOT / "FIGURE3_QA_GATE.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
