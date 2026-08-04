from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS = Path(r"F:\ACM\experiments\phase3a\formal\metrics_summary.json")
PREDICTIONS = Path(r"F:\ACM\experiments\phase3a\formal\oof_predictions.csv")
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


def km_event_probability(times: pd.Series, events: pd.Series, horizon: float) -> float:
    frame = pd.DataFrame({"time": times.astype(float), "event": events.astype(int)}).sort_values("time")
    survival = 1.0
    for time in sorted(frame.loc[(frame["time"] <= horizon) & frame["event"].eq(1), "time"].unique()):
        at_risk = int((frame["time"] >= time).sum())
        deaths = int(((frame["time"] == time) & frame["event"].eq(1)).sum())
        if at_risk > 0:
            survival *= 1.0 - deaths / at_risk
    return float(1.0 - survival)


def main() -> int:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTIONS)
    predictions["model_short"] = predictions["model"].map(MODEL_MAP)
    auc_rows = []
    for model_key, model_short in MODEL_MAP.items():
        block = metrics["metrics"][model_key]
        for horizon in (12, 36, 60):
            for index, value in enumerate(block[f"auc_{horizon}m"]["per_fold"]):
                auc_rows.append(
                    {
                        "model": model_key,
                        "model_short": model_short,
                        "repeat": index // 5 + 1,
                        "fold": index % 5 + 1,
                        "horizon_months": horizon,
                        "auc": value,
                    }
                )
    auc = pd.DataFrame(auc_rows)
    auc_dir = ROOT / "panel_a_auc_trajectories"
    auc_dir.mkdir(parents=True, exist_ok=True)
    auc.to_csv(auc_dir / "source_data.csv", index=False)

    horizon_column = {
        12: "survival_probability_12m",
        36: "survival_probability_36m",
        60: "survival_probability_60m",
    }
    output_counts = {}
    for horizon, folder in [
        (12, "panel_b_calibration_12m"),
        (36, "panel_c_calibration_36m"),
        (60, "panel_d_calibration_60m"),
    ]:
        rows = []
        for model_short in ["M1", "M4"]:
            for repeat in range(1, 6):
                subset = predictions[
                    predictions["model_short"].eq(model_short)
                    & predictions["repeat"].eq(repeat)
                ].copy()
                subset["predicted_event_risk"] = 1.0 - subset[horizon_column[horizon]]
                subset["risk_bin"] = pd.qcut(
                    subset["predicted_event_risk"].rank(method="first"),
                    6,
                    labels=False,
                ).astype(int) + 1
                for risk_bin, group in subset.groupby("risk_bin", sort=True):
                    rows.append(
                        {
                            "model_short": model_short,
                            "repeat": repeat,
                            "horizon_months": horizon,
                            "risk_bin": int(risk_bin),
                            "n_patients": len(group),
                            "mean_predicted_event_risk": float(group["predicted_event_risk"].mean()),
                            "observed_event_probability_km": km_event_probability(
                                group["survival_months"], group["event"], horizon
                            ),
                        }
                    )
        frame = pd.DataFrame(rows)
        out = ROOT / folder
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "source_data.csv", index=False)
        output_counts[folder] = len(frame)

    checks = {
        "formal_predictions_9075": len(predictions) == 9075,
        "five_models": predictions["model_short"].nunique() == 5,
        "five_repeats": predictions["repeat"].nunique() == 5,
        "363_patients_per_model_repeat": bool(
            (predictions.groupby(["model_short", "repeat"]).size() == 363).all()
        ),
        "auc_rows_375": len(auc) == 375,
        "calibration_60_rows_each": all(value == 60 for value in output_counts.values()),
        "calibration_values_bounded": True,
    }
    for folder in output_counts:
        frame = pd.read_csv(ROOT / folder / "source_data.csv")
        checks["calibration_values_bounded"] = checks["calibration_values_bounded"] and bool(
            frame["mean_predicted_event_risk"].between(0, 1).all()
            and frame["observed_event_probability_km"].between(0, 1).all()
        )
    report = {
        "status": "SUPP_FIGURE_S1_SOURCE_READY" if all(checks.values()) else "SUPP_FIGURE_S1_SOURCE_FAILED",
        "success": bool(all(checks.values())),
        "checks": checks,
        "inputs": {str(METRICS): sha256(METRICS), str(PREDICTIONS): sha256(PREDICTIONS)},
        "rows": {"auc": len(auc), **output_counts},
    }
    (ROOT / "SOURCE_DATA_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
