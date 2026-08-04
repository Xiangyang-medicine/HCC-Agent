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

from supp_s2_style import MID, MODEL_COLORS, panel_heading, quiet_axis, set_style  # noqa: E402


SOURCE = Path(__file__).with_name("source_data.csv")


def draw(ax: plt.Axes, letter: str = "a") -> None:
    data = pd.read_csv(SOURCE)
    models = ["M1", "M2", "M3", "M4", "M5"]
    panel_heading(
        ax,
        letter,
        "Repeat-to-repeat risk-rank agreement",
        "Ten pairwise Spearman correlations per model; OOF percentile ranks",
    )
    quiet_axis(ax)
    values = [
        data.loc[data["model_short"].eq(model), "spearman_rho"].to_numpy(float)
        for model in models
    ]
    boxes = ax.boxplot(
        values,
        positions=np.arange(5),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#20262E", "linewidth": 1.0},
        whiskerprops={"color": MID, "linewidth": 0.75},
        capprops={"color": MID, "linewidth": 0.75},
    )
    for patch, model in zip(boxes["boxes"], models):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_edgecolor(MODEL_COLORS[model])
        patch.set_alpha(0.17)
    for index, (model, series) in enumerate(zip(models, values)):
        jitter = (np.arange(len(series)) - (len(series) - 1) / 2) * 0.032
        ax.scatter(
            np.full(len(series), index) + jitter,
            series,
            s=10,
            color=MODEL_COLORS[model],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
        ax.text(
            index,
            -0.07,
            f"{np.median(series):.2f}",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color=MODEL_COLORS[model],
            fontweight="bold",
        )
    ax.axhline(0, color=MID, linestyle=(0, (3, 2)), linewidth=0.75)
    ax.set_xticks(range(5), models)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-0.10, 0.98)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    set_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    draw(ax)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.82, bottom=0.17)
    stem = Path(__file__).with_name("panel_a_repeat_correlation")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
