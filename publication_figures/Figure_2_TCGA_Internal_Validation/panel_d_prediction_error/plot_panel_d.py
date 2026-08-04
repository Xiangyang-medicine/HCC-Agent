from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import LIGHT, MID, MODEL_COLORS, panel_heading, quiet_axis, save_publication_figure, set_style


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "d") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    ibs = json.loads((HERE / "ibs_summary.json").read_text(encoding="utf-8"))
    models = ["M1", "M4"]
    offsets = {"M1": -0.13, "M4": 0.13}

    for horizon_index, horizon in enumerate([12, 36, 60], start=1):
        wide = (
            data.loc[data["horizon_months"].eq(horizon)]
            .pivot(index=["repeat", "fold"], columns="model_short", values="brier_score")
            .sort_index()
        )
        for _, row in wide.iterrows():
            ax.plot(
                [horizon_index + offsets["M1"], horizon_index + offsets["M4"]],
                [row["M1"], row["M4"]],
                color=LIGHT,
                linewidth=0.55,
                alpha=0.8,
                zorder=1,
            )
        for model in models:
            values = wide[model].to_numpy()
            x = horizon_index + offsets[model]
            ax.scatter(
                np.full(len(values), x),
                values,
                s=7.0,
                color=MODEL_COLORS[model],
                alpha=0.47,
                edgecolor="none",
                zorder=2,
            )
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            ax.errorbar(
                x,
                mean,
                yerr=1.96 * se,
                fmt="o",
                markersize=4.0,
                color=MODEL_COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.35,
                capsize=2.0,
                linewidth=1.0,
                zorder=4,
            )

    ax.set_xlim(0.55, 3.45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["12", "36", "60"])
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("Brier score")
    ax.text(
        0.02,
        0.98,
        (
            f"Mean IBS\n"
            f"M1  {ibs['M1_clinical_cox']:.3f}\n"
            f"M4  {ibs['M4_combined_rsf']:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=5.5,
        ha="left",
        va="top",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": LIGHT, "linewidth": 0.55},
    )
    panel_heading(
        ax,
        letter,
        "Prediction error across horizons",
        "25 paired outer folds · lower is better",
    )
    ax.plot([], [], color=MODEL_COLORS["M1"], marker="o", linestyle="", label="M1 clinical Cox")
    ax.plot([], [], color=MODEL_COLORS["M4"], marker="o", linestyle="", label="M4 combined RSF")
    ax.legend(loc="lower right")
    quiet_axis(ax, grid_axis="y")


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    draw(ax)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.87, bottom=0.20)
    save_publication_figure(fig, HERE / "panel_d_prediction_error")
    plt.close(fig)


if __name__ == "__main__":
    main()
