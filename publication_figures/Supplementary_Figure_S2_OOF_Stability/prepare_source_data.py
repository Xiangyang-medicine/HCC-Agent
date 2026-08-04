from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
INPUT = Path(r"F:\ACM\experiments\phase3a\formal\oof_predictions.csv")
MODEL_MAP = {
    "M1_clinical_cox": "M1",
    "M2_gene_elasticnet": "M2",
    "M3_combined_elasticnet": "M3",
    "M4_combined_rsf": "M4",
    "M5_deepsurv": "M5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quintiles(values: pd.Series) -> pd.Series:
    return pd.qcut(values.rank(method="first"), 5, labels=False).astype(int) + 1


def main() -> int:
    data = pd.read_csv(INPUT)
    required = {
        "case_id",
        "repeat",
        "fold",
        "model",
        "risk_score",
        "survival_probability_12m",
        "survival_probability_36m",
        "survival_probability_60m",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data["model_short"] = data["model"].map(MODEL_MAP)
    if data["model_short"].isna().any():
        raise ValueError("Unexpected model identifier")
    case_map = {
        case_id: f"P{index:03d}"
        for index, case_id in enumerate(sorted(data["case_id"].unique()), start=1)
    }
    data["case_label"] = data["case_id"].map(case_map)
    data["risk_percentile"] = data.groupby(["model_short", "repeat"])[
        "risk_score"
    ].rank(pct=True, method="average")
    data["risk_quintile"] = data.groupby(["model_short", "repeat"])[
        "risk_score"
    ].transform(quintiles)

    pair_rows: list[dict[str, float | int | str]] = []
    stability_rows: list[dict[str, float | str]] = []
    density_rows: list[dict[str, float | int | str]] = []
    consensus_rows: list[dict[str, float | int | str]] = []
    qc_rows: list[dict[str, float | int | str]] = []

    for model in ["M1", "M2", "M3", "M4", "M5"]:
        subset = data[data["model_short"].eq(model)].copy()
        percentiles = subset.pivot(
            index="case_label", columns="repeat", values="risk_percentile"
        ).sort_index()
        quintile_matrix = subset.pivot(
            index="case_label", columns="repeat", values="risk_quintile"
        ).sort_index()
        for repeat_a in range(1, 6):
            for repeat_b in range(repeat_a + 1, 6):
                rho = float(
                    spearmanr(
                        percentiles[repeat_a].to_numpy(),
                        percentiles[repeat_b].to_numpy(),
                    ).statistic
                )
                pair_rows.append(
                    {
                        "model_short": model,
                        "repeat_a": repeat_a,
                        "repeat_b": repeat_b,
                        "spearman_rho": rho,
                    }
                )
                if model in {"M4", "M5"}:
                    for case_label in percentiles.index:
                        density_rows.append(
                            {
                                "model_short": model,
                                "case_label": case_label,
                                "repeat_a": repeat_a,
                                "repeat_b": repeat_b,
                                "risk_percentile_a": float(percentiles.loc[case_label, repeat_a]),
                                "risk_percentile_b": float(percentiles.loc[case_label, repeat_b]),
                            }
                        )
        for case_label, row in percentiles.iterrows():
            stability_rows.append(
                {
                    "model_short": model,
                    "case_label": case_label,
                    "mean_risk_percentile": float(row.mean()),
                    "risk_percentile_sd": float(row.std(ddof=1)),
                }
            )
        modal_counts = quintile_matrix.apply(
            lambda row: int(row.value_counts().max()), axis=1
        )
        distribution = modal_counts.value_counts().reindex(range(1, 6), fill_value=0)
        for modal_count, patient_count in distribution.items():
            consensus_rows.append(
                {
                    "model_short": model,
                    "modal_repeat_count": int(modal_count),
                    "patient_count": int(patient_count),
                    "fraction": float(patient_count / len(modal_counts)),
                }
            )
        probabilities = subset[
            [
                "survival_probability_12m",
                "survival_probability_36m",
                "survival_probability_60m",
            ]
        ]
        bounded = probabilities.ge(0).all(axis=1) & probabilities.le(1).all(axis=1)
        monotonic = (
            probabilities["survival_probability_12m"]
            >= probabilities["survival_probability_36m"]
        ) & (
            probabilities["survival_probability_36m"]
            >= probabilities["survival_probability_60m"]
        )
        qc_rows.append(
            {
                "model_short": model,
                "prediction_rows": len(subset),
                "bounded_rows": int(bounded.sum()),
                "monotonic_rows": int(monotonic.sum()),
                "bounded_fraction": float(bounded.mean()),
                "monotonic_fraction": float(monotonic.mean()),
            }
        )

    pair_data = pd.DataFrame(pair_rows)
    stability_data = pd.DataFrame(stability_rows)
    density = pd.DataFrame(density_rows)
    consensus = pd.DataFrame(consensus_rows)
    qc = pd.DataFrame(qc_rows)

    outputs = {
        ROOT / "panel_a_repeat_correlation" / "source_data.csv": pair_data,
        ROOT / "panel_b_patient_rank_dispersion" / "source_data.csv": stability_data,
        ROOT / "panel_c_m4_quintile_transition" / "source_data.csv": density[
            density["model_short"].eq("M4")
        ],
        ROOT / "panel_d_m5_quintile_transition" / "source_data.csv": density[
            density["model_short"].eq("M5")
        ],
        ROOT / "panel_e_consensus_and_qc" / "source_data.csv": consensus,
        ROOT / "panel_e_consensus_and_qc" / "source_data_qc.csv": qc,
    }
    for path, frame in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    checks = {
        "input_rows_9075": len(data) == 9075,
        "five_models": sorted(data["model_short"].unique()) == ["M1", "M2", "M3", "M4", "M5"],
        "363_patients": data["case_id"].nunique() == 363,
        "five_repeats": sorted(data["repeat"].unique()) == [1, 2, 3, 4, 5],
        "no_duplicate_model_repeat_patient": not data.duplicated(
            ["model_short", "repeat", "case_id"]
        ).any(),
        "pair_rows_50": len(pair_data) == 50,
        "stability_rows_1815": len(stability_data) == 1815,
        "density_rows_m4_3630": len(outputs[ROOT / "panel_c_m4_quintile_transition" / "source_data.csv"]) == 3630,
        "density_rows_m5_3630": len(outputs[ROOT / "panel_d_m5_quintile_transition" / "source_data.csv"]) == 3630,
        "consensus_rows_25": len(consensus) == 25,
        "qc_rows_5": len(qc) == 5,
        "all_probabilities_bounded": bool((qc["bounded_fraction"] == 1).all()),
        "all_survival_curves_monotonic": bool((qc["monotonic_fraction"] == 1).all()),
    }
    report = {
        "status": "SUPP_FIGURE_S2_SOURCE_READY" if all(checks.values()) else "SUPP_FIGURE_S2_SOURCE_FAILED",
        "success": bool(all(checks.values())),
        "checks": checks,
        "canonical_input": str(INPUT),
        "canonical_input_sha256": sha256(INPUT),
        "output_rows": {str(path.relative_to(ROOT)): len(frame) for path, frame in outputs.items()},
    }
    (ROOT / "SOURCE_DATA_GATE.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
