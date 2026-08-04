from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supp_s2_style import MID, PALE_TEAL, TEAL, TEXT, panel_heading, set_style  # noqa: E402


SOURCE = Path(__file__).with_name("source_data.csv")
QC_SOURCE = Path(__file__).with_name("source_data_qc.csv")
COLORS = {1: "#EEF1F3", 2: "#D8DEE3", 3: "#B7C2CB", 4: "#72B8AA", 5: TEAL}


def draw(ax: plt.Axes, letter: str = "e") -> None:
    data = pd.read_csv(SOURCE)
    qc = pd.read_csv(QC_SOURCE)
    models = ["M1", "M2", "M3", "M4", "M5"]
    panel_heading(
        ax,
        letter,
        "Consensus risk-quintile assignment across five repeats",
        "Stacked fractions show the number of repeats matching each patient's modal quintile",
    )
    y = np.arange(len(models))
    left = np.zeros(len(models))
    for modal_count in range(1, 6):
        values = (
            data[data["modal_repeat_count"].eq(modal_count)]
            .set_index("model_short")
            .reindex(models)["fraction"]
            .fillna(0)
            .to_numpy(float)
        )
        ax.barh(y, values, left=left, color=COLORS[modal_count], edgecolor="white", linewidth=0.45, height=0.68, label=str(modal_count))
        for index, value in enumerate(values):
            if value >= 0.075:
                ax.text(left[index] + value / 2, index, f"{100 * value:.0f}%", ha="center", va="center", fontsize=5.0, color="white" if modal_count >= 4 else TEXT, fontweight="bold" if modal_count >= 4 else "normal")
        left += values
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of patients")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDE3E8", linewidth=0.5, zorder=0)
    ax.legend(title="Repeats matching modal quintile", ncol=5, loc="upper center", bbox_to_anchor=(0.44, -0.27), frameon=False, columnspacing=0.8, handlelength=1.4)

    total_rows = int(qc["prediction_rows"].sum())
    bounded_rows = int(qc["bounded_rows"].sum())
    monotonic_rows = int(qc["monotonic_rows"].sum())
    badge = FancyBboxPatch((1.025, 0.24), 0.34, 0.48, transform=ax.transAxes, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=PALE_TEAL, edgecolor=TEAL, linewidth=0.75, clip_on=False)
    ax.add_patch(badge)
    ax.text(1.195, 0.48, f"Structural prediction QC\n{bounded_rows:,}/{total_rows:,} bounded\n{monotonic_rows:,}/{total_rows:,} monotonic\nS12 ≥ S36 ≥ S60", transform=ax.transAxes, ha="center", va="center", fontsize=5.4, color=TEXT, fontweight="bold")
    ax.text(0.0, -0.12, "Percentages <7.5% are not printed inside segments.", transform=ax.transAxes, fontsize=5.0, color=MID, ha="left")


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 6.6, "svg.fonttype": "none", "pdf.fonttype": 42})
    set_style()
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    draw(ax)
    fig.subplots_adjust(left=0.12, right=0.78, top=0.78, bottom=0.26)
    stem = Path(__file__).with_name("panel_e_consensus_and_qc")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
