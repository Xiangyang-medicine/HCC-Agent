from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_v2_style import (  # noqa: E402
    B2_COLOR,
    B4_COLOR,
    MID,
    panel_label,
    quiet_axis,
    save_publication_figure,
    set_style,
)


SOURCE = Path(__file__).with_name("source_data.csv")


def draw(ax: plt.Axes, letter: str = "d") -> None:
    data = pd.read_csv(SOURCE)
    metrics = [
        "exact_extractive_claim_support",
        "retrieved_passage_citation_validity",
        "exact_three_run_agreement",
    ]
    labels = [
        "Exact extractive\nsupport",
        "Assigned-passage\ncitation validity",
        "Exact three-run\nagreement",
    ]
    systems = [
        ("B2_SINGLE_LLM_WITH_TOOLS", "B2", B2_COLOR, -0.10),
        ("B4_FULL_CLOSED_LOOP", "B4", B4_COLOR, 0.10),
    ]
    y = np.arange(len(metrics))[::-1]
    panel_label(ax, letter)
    ax.set_title(
        "Traceability and repeatability",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    quiet_axis(ax, grid_axis="x")

    for system, short, color, offset in systems:
        subset = data[data["system"].eq(system)].set_index("metric")
        for index, metric in enumerate(metrics):
            row = subset.loc[metric]
            value = row["value"] * 100
            low = row["ci_low"] * 100
            high = row["ci_high"] * 100
            ax.errorbar(
                value,
                y[index] + offset,
                xerr=[[value - low], [high - value]],
                fmt="o",
                markersize=5.8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                ecolor=color,
                elinewidth=1.2,
                capsize=2.4,
                label=short if index == 0 else None,
                zorder=3,
            )
            label_above = short == "B4" or (short == "B2" and index == 2)
            ax.text(
                value,
                y[index] + offset + (0.12 if label_above else -0.12),
                f"{short} {value:.1f}",
                fontsize=5.2,
                color=color,
                ha="center",
                va="bottom" if label_above else "top",
            )
    ax.set_yticks([])
    for index, label in enumerate(labels):
        ax.text(
            43.0,
            y[index],
            label,
            ha="left",
            va="center",
            fontsize=5.7,
            color=MID,
        )
    ax.set_xlim(42, 102)
    ax.set_xlabel("Rate (%)")


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.9, 2.5))
    draw(ax)
    fig.subplots_adjust(left=0.32, right=0.98, top=0.89, bottom=0.22)
    save_publication_figure(
        fig, Path(__file__).with_name("panel_d_traceability_reliability")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
