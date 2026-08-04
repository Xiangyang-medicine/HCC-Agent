from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import (  # noqa: E402
    SYSTEM_COLORS,
    clean_axis,
    panel_label,
    save_publication_figure,
    set_style,
)


SYSTEMS = [
    "B1_SINGLE_LLM_NO_TOOLS",
    "B2_SINGLE_LLM_WITH_TOOLS",
    "B3_MULTI_AGENT_NO_VERIFIER",
    "B4_FULL_CLOSED_LOOP",
]
METRICS = [
    "supported_claim_precision",
    "citation_correctness",
    "unsupported_claim_rate",
]


def draw(ax: plt.Axes, letter: str | None = None) -> None:
    data = pd.read_csv(Path(__file__).resolve().parent / "source_data.csv")
    x = np.arange(len(METRICS))
    width = 0.19
    offsets = (np.arange(len(SYSTEMS)) - 1.5) * width
    short = {
        "B1_SINGLE_LLM_NO_TOOLS": "B1",
        "B2_SINGLE_LLM_WITH_TOOLS": "B2",
        "B3_MULTI_AGENT_NO_VERIFIER": "B3",
        "B4_FULL_CLOSED_LOOP": "B4",
    }
    for offset, system in zip(offsets, SYSTEMS):
        frame = data[data["system"].eq(system)].set_index("metric").loc[METRICS]
        values = frame["value"].to_numpy() * 100
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=SYSTEM_COLORS[system],
            label=short[system],
            zorder=2,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.6,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=5.2,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["Supported\nclaims", "Correct\ncitations", "Unsupported\nclaims"])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 119)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title("Evidence grounding and safety", loc="left", fontweight="bold", pad=5)
    clean_axis(ax, "y")
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.8,
        handlelength=1.4,
    )
    if letter:
        panel_label(ax, letter, x=-0.16, y=1.10)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    draw(ax, "c")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.82)
    save_publication_figure(fig, Path(__file__).resolve().parent / "panel_c")
    plt.close(fig)


if __name__ == "__main__":
    main()
