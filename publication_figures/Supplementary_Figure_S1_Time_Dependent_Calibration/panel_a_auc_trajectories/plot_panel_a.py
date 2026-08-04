from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supp_s1_style import MID, MODEL_COLORS, panel_heading, quiet_axis, set_style  # noqa: E402


SOURCE = Path(__file__).with_name("source_data.csv")


def draw(ax: plt.Axes, letter: str = "a") -> None:
    data = pd.read_csv(SOURCE)
    panel_heading(
        ax,
        letter,
        "Time-dependent discrimination trajectories",
        "All 25 outer test folds shown lightly; lines connect fold means",
    )
    quiet_axis(ax, "y")
    horizons = [12, 36, 60]
    models = ["M1", "M2", "M3", "M4", "M5"]
    offsets = np.linspace(-0.16, 0.16, len(models))
    x = np.arange(3, dtype=float)
    for model, offset in zip(models, offsets):
        subset = data[data["model_short"].eq(model)]
        means = []
        sds = []
        for index, horizon in enumerate(horizons):
            values = subset.loc[subset["horizon_months"].eq(horizon), "auc"].to_numpy(float)
            means.append(values.mean())
            sds.append(values.std(ddof=1))
            jitter = (np.arange(len(values)) % 5 - 2) * 0.006
            ax.scatter(np.full(len(values), x[index] + offset) + jitter, values, s=5, color=MODEL_COLORS[model], alpha=0.18, edgecolors="none", zorder=2)
        ax.errorbar(
            x + offset,
            means,
            yerr=sds,
            color=MODEL_COLORS[model],
            marker="o",
            markersize=4.8 if model == "M4" else 4.0,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.55 if model == "M4" else 0.95,
            capsize=2.2,
            zorder=4 if model == "M4" else 3,
            label=model,
        )
    ax.axhline(0.5, color=MID, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(2.42, 0.505, "chance", fontsize=5.1, color=MID, ha="right", va="bottom")
    ax.set_xticks(x, ["12 months", "36 months", "60 months"])
    ax.set_ylabel("Time-dependent AUC")
    ax.set_xlim(-0.43, 2.47)
    ax.set_ylim(0.15, 0.93)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), columnspacing=0.9, handlelength=1.4)


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 6.6, "svg.fonttype": "none", "pdf.fonttype": 42})
    set_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    draw(ax)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.81, bottom=0.25)
    stem = Path(__file__).with_name("panel_a_auc_trajectories")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
