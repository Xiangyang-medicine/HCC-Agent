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


def draw(ax: plt.Axes, letter: str = "b") -> None:
    data = pd.read_csv(SOURCE)
    models = ["M1", "M2", "M3", "M4", "M5"]
    panel_heading(
        ax,
        letter,
        "Patient-level rank dispersion",
        "Within-patient SD of five OOF percentile ranks; lower is more stable",
    )
    quiet_axis(ax)
    values = [
        data.loc[data["model_short"].eq(model), "risk_percentile_sd"].to_numpy(float)
        for model in models
    ]
    violin = ax.violinplot(
        values,
        positions=np.arange(5),
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, model in zip(violin["bodies"], models):
        body.set_facecolor(MODEL_COLORS[model])
        body.set_edgecolor(MODEL_COLORS[model])
        body.set_alpha(0.18)
        body.set_linewidth(0.6)
    for index, (model, series) in enumerate(zip(models, values)):
        q1, median, q3 = np.quantile(series, [0.25, 0.5, 0.75])
        ax.vlines(index, q1, q3, color=MODEL_COLORS[model], linewidth=4.2, alpha=0.75)
        ax.scatter(index, median, s=19, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.45, zorder=4)
        ax.text(
            index,
            0.445,
            f"{median:.2f}",
            ha="center",
            va="top",
            fontsize=5.2,
            color=MODEL_COLORS[model],
            fontweight="bold",
        )
    ax.set_xticks(range(5), models)
    ax.set_ylabel("SD of risk percentile")
    ax.set_ylim(0, 0.46)


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
    fig.subplots_adjust(left=0.16, right=0.98, top=0.82, bottom=0.17)
    stem = Path(__file__).with_name("panel_b_patient_rank_dispersion")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
