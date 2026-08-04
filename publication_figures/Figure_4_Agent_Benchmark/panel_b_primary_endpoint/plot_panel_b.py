from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure4_style import (  # noqa: E402
    B2_COLOR,
    B2_LIGHT,
    B4_COLOR,
    B4_LIGHT,
    LIGHT,
    MID,
    TEXT,
    panel_heading,
    quiet_axis,
    save_publication_figure,
    set_style,
)


SOURCE = Path(__file__).with_name("source_data.csv")
COMPARISON = Path(__file__).with_name("paired_comparison.csv")


def draw(ax: plt.Axes, letter: str = "b") -> None:
    data = pd.read_csv(SOURCE)
    data = data[data["analysis_status"].eq("prespecified_confirmatory")].copy()
    comparison = pd.read_csv(COMPARISON).iloc[0]
    panel_heading(
        ax,
        letter,
        "Closed-loop execution improves task completion",
        "Prespecified external composite pass · 100 cases × 3 runs",
    )
    quiet_axis(ax, grid_axis="x")

    order = [
        "B2_SINGLE_LLM_WITH_TOOLS",
        "B4_FULL_CLOSED_LOOP",
    ]
    labels = ["B2  Single controller", "B4  Closed loop"]
    colors = [B2_COLOR, B4_COLOR]
    y = np.array([0.55, 1.05])
    for index, (system, label, color) in enumerate(zip(order, labels, colors)):
        row = data[data["system"].eq(system)].iloc[0]
        rate = float(row["rate"]) * 100
        low = float(row["ci_low"]) * 100
        high = float(row["ci_high"]) * 100
        ax.errorbar(
            rate,
            y[index],
            xerr=[[rate - low], [high - rate]],
            fmt="o",
            markersize=7.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.2,
            capthick=1.1,
            zorder=4,
        )
        ax.text(
            rate,
            y[index] + 0.13,
            f"{rate:.1f}%  ({int(row['successes'])}/300)",
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold" if system.endswith("FULL_CLOSED_LOOP") else "normal",
            color=color,
        )
        ax.text(
            66.0,
            y[index],
            label,
            ha="left",
            va="center",
            fontsize=5.7,
            color=color,
        )
    ax.set_yticks([])
    ax.set_xlim(65, 104)
    ax.set_ylim(-0.40, 1.68)
    ax.set_xlabel("Frozen external composite pass (%)")
    ax.text(
        65.2,
        1.50,
        (
            f"Δ +{comparison['difference'] * 100:.1f} pp  "
            f"(95% CI {comparison['ci_low'] * 100:.1f} to "
            f"{comparison['ci_high'] * 100:.1f})\n"
            f"paired sign-permutation  p={comparison['p_value']:.1e}"
        ),
        ha="left",
        va="top",
        fontsize=6.15,
        color=TEXT,
    )
    ax.text(
        65.2,
        -0.28,
        "Post-hoc strict structural audit changed 0/600 B2–B4 clean runs.",
        fontsize=5.0,
        color=MID,
    )

    inset = ax.inset_axes([0.72, 0.03, 0.26, 0.36])
    matrix = np.array(
        [
            [comparison["both_pass"], comparison["b4_only_pass"]],
            [comparison["b2_only_pass"], comparison["both_fail"]],
        ],
        dtype=int,
    )
    fills = [[B4_LIGHT, B4_COLOR], [B2_COLOR, LIGHT]]
    inset.set_xlim(0, 2)
    inset.set_ylim(0, 2)
    for row in range(2):
        for col in range(2):
            y0 = 1 - row
            inset.add_patch(
                Rectangle(
                    (col, y0),
                    1,
                    1,
                    facecolor=fills[row][col],
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            dark = (row, col) in {(0, 1), (1, 0)}
            inset.text(
                col + 0.5,
                y0 + 0.5,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=6.4,
                fontweight="bold",
                color="white" if dark else TEXT,
            )
    inset.set_xticks([0.5, 1.5])
    inset.set_xticklabels(["B2 pass", "B2 fail"], fontsize=4.9)
    inset.set_yticks([1.5, 0.5])
    inset.set_yticklabels(["B4 pass", "B4 fail"], fontsize=4.9)
    inset.tick_params(length=0, pad=1)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_title("Paired runs", fontsize=5.2, pad=1.5)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    draw(ax)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.90, bottom=0.22)
    save_publication_figure(fig, Path(__file__).with_name("panel_b_primary_endpoint"))
    plt.close(fig)


if __name__ == "__main__":
    main()
