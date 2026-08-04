from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from supp_s1_style import MID, MODEL_COLORS, panel_heading, quiet_axis


def draw_calibration(
    ax: plt.Axes,
    source: Path,
    letter: str,
    horizon: int,
) -> None:
    data = pd.read_csv(source)
    panel_heading(
        ax,
        letter,
        f"{horizon}-month OOF calibration",
        "Six risk bins per repeat; observed risk from Kaplan–Meier",
    )
    quiet_axis(ax, "both")
    ax.plot([0, 1], [0, 1], color=MID, linestyle=(0, (3, 2)), linewidth=0.85, zorder=1)
    for model in ["M1", "M4"]:
        model_data = data[data["model_short"].eq(model)]
        for repeat in range(1, 6):
            curve = model_data[model_data["repeat"].eq(repeat)].sort_values("mean_predicted_event_risk")
            ax.plot(
                curve["mean_predicted_event_risk"],
                curve["observed_event_probability_km"],
                color=MODEL_COLORS[model],
                alpha=0.17,
                linewidth=0.75,
                zorder=2,
            )
        summary = (
            model_data.groupby("risk_bin", as_index=False)
            .agg(
                predicted=("mean_predicted_event_risk", "mean"),
                observed=("observed_event_probability_km", "mean"),
                observed_sd=("observed_event_probability_km", "std"),
            )
            .sort_values("predicted")
        )
        ax.errorbar(
            summary["predicted"],
            summary["observed"],
            yerr=summary["observed_sd"],
            color=MODEL_COLORS[model],
            marker="o",
            markersize=4.2 if model == "M4" else 3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.45 if model == "M4" else 1.05,
            capsize=2.0,
            zorder=4,
            label=model,
        )
    ax.set_xlim(0, 0.92)
    ax.set_ylim(0, 0.92)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted event probability")
    ax.set_ylabel("Kaplan–Meier observed event probability")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.02, "n=363 per repeat", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=MID)
