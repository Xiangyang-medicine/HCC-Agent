from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure3_style import (  # noqa: E402
    BLUE,
    MID,
    ORANGE,
    TEXT,
    panel_heading,
    quiet_axis,
    save_publication_figure,
    set_style,
)


SOURCE = Path(__file__).with_name("source_data.csv")


def draw(ax: plt.Axes, letter: str = "b") -> None:
    data = pd.read_csv(SOURCE)
    order = [("GSE14520", "GPL3921"), ("GSE116174", "GPL570")]
    data = data.set_index(["cohort", "platform"]).loc[order].reset_index()
    y = [1.15, 0.45]
    colors = [ORANGE, BLUE]

    panel_heading(
        ax,
        letter,
        "Continuous score and overall survival",
        "Hazard ratio per 1-SD higher score · no external recalibration",
        label_x=-0.10,
    )
    quiet_axis(ax, grid_axis="x")
    ax.axvline(1.0, color=MID, linewidth=0.8, linestyle="--", zorder=1)

    for index, row in enumerate(data.itertuples(index=False)):
        ax.errorbar(
            row.hazard_ratio_per_1sd,
            y[index],
            xerr=[
                [row.hazard_ratio_per_1sd - row.ci_low],
                [row.ci_high - row.hazard_ratio_per_1sd],
            ],
            fmt="o",
            markersize=6.0,
            markerfacecolor=colors[index],
            markeredgecolor="white",
            markeredgewidth=0.7,
            ecolor=colors[index],
            elinewidth=1.3,
            capsize=3.0,
            zorder=3,
        )
        p_text = "<0.001" if row.wald_p < 0.001 else f"={row.wald_p:.3f}"
        ax.text(
            2.47,
            y[index] + 0.10,
            (
                f"HR {row.hazard_ratio_per_1sd:.2f} "
                f"[{row.ci_low:.2f}–{row.ci_high:.2f}]"
            ),
            fontsize=5.0,
            color=TEXT,
            ha="right",
            va="center",
        )
        ax.text(
            2.47,
            y[index] - 0.12,
            f"N={row.n}, events={row.events}; P{p_text}",
            fontsize=5.0,
            color=MID,
            ha="right",
            va="center",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(["GSE14520\nGPL3921", "GSE116174\nGPL570"])
    ax.set_xlim(0.70, 2.50)
    ax.set_ylim(0.05, 1.55)
    ax.set_xlabel("Hazard ratio per 1-SD higher risk score (95% CI)")
    ax.text(1.02, 1.49, "null", fontsize=5.0, color=MID, va="top")
    ax.text(
        0.99,
        0.02,
        "Continuous score; no cutpoint or external recalibration",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    draw(ax)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.88, bottom=0.20)
    save_publication_figure(
        fig, Path(__file__).with_name("panel_b_cohort_flow")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
