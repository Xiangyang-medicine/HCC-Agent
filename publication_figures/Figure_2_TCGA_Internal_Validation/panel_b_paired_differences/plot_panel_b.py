from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import BLUE, LIGHT, MID, MODEL_COLORS, RUST, TEXT, panel_heading, quiet_axis, save_publication_figure, set_style


DATA = Path(__file__).resolve().parent / "source_data.csv"


def _p_text(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def draw(ax: plt.Axes, letter: str = "b") -> None:
    data = pd.read_csv(DATA)
    order = [
        ("Harrell C", "M3 vs M1"),
        ("Harrell C", "M4 vs M1"),
        ("Harrell C", "M5 vs M1"),
        ("Harrell C", "M3 vs M2"),
        ("Uno C", "M3 vs M1"),
        ("Uno C", "M4 vs M1"),
        ("Uno C", "M5 vs M1"),
        ("Uno C", "M3 vs M2"),
    ]
    y_positions = [8.4, 7.4, 6.4, 5.4, 3.7, 2.7, 1.7, 0.7]

    ax.axvspan(-0.22, 0, color="#F7F0ED", alpha=0.65, zorder=0)
    ax.axvspan(0, 0.23, color="#EFF6F4", alpha=0.65, zorder=0)
    ax.axvline(0, color=TEXT, linewidth=0.8, zorder=1)
    ax.axhline(4.55, color=LIGHT, linewidth=0.8)

    for (metric, comparison), y in zip(order, y_positions):
        row = data.loc[
            data["metric_label"].eq(metric)
            & data["comparison_label"].eq(comparison)
        ].iloc[0]
        model = comparison.split()[0]
        color = MODEL_COLORS.get(model, BLUE)
        if model == "M5":
            color = RUST
        ax.plot(
            [row["ci_lower"], row["ci_upper"]],
            [y, y],
            color=color,
            linewidth=1.4,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            row["mean_diff"],
            y,
            marker="o",
            markersize=4.4,
            markerfacecolor=color if bool(row["significant_adjusted"]) else "white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=4,
        )
        ax.text(
            -0.227,
            y,
            comparison,
            fontsize=5.7,
            ha="right",
            va="center",
            color=TEXT,
            clip_on=False,
        )
        ax.text(
            0.242,
            y,
            _p_text(float(row["p_value_adjusted"])),
            fontsize=5.6,
            fontweight="bold" if bool(row["significant_adjusted"]) else "normal",
            ha="right",
            va="center",
            color=color if bool(row["significant_adjusted"]) else MID,
            clip_on=True,
        )

    ax.text(-0.227, 9.05, "Comparison", fontsize=5.4, fontweight="bold", ha="right")
    ax.text(0.242, 9.05, r"Adjusted $P$", fontsize=5.4, fontweight="bold", ha="right")
    ax.text(-0.217, 8.95, "Harrell C", fontsize=6.1, fontweight="bold", color=TEXT)
    ax.text(-0.217, 4.25, "Uno C (IPCW)", fontsize=6.1, fontweight="bold", color=TEXT)
    ax.text(
        -0.12,
        -0.05,
        "favours comparator",
        fontsize=5.2,
        color=MID,
        ha="center",
        va="top",
    )
    ax.text(
        0.12,
        -0.05,
        "favours model A",
        fontsize=5.2,
        color=MID,
        ha="center",
        va="top",
    )
    ax.set_xlim(-0.22, 0.25)
    ax.set_ylim(0.0, 9.45)
    ax.set_yticks([])
    ax.set_xlabel("Paired mean difference (95% bootstrap CI)")
    panel_heading(
        ax,
        letter,
        "Pre-specified paired comparisons",
        "Patient-level bootstrap · 1,000 resamples · Bonferroni correction",
        label_x=-0.22,
    )
    quiet_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    draw(ax)
    fig.subplots_adjust(left=0.31, right=0.84, top=0.88, bottom=0.17)
    save_publication_figure(
        fig, Path(__file__).resolve().parent / "panel_b_paired_differences"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
