from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure4_style import (  # noqa: E402
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


def p_label(value: float) -> str:
    return "pHolm<0.001" if value < 0.001 else f"pHolm={value:.3f}"


def draw(ax: plt.Axes, letter: str = "c") -> None:
    data = pd.read_csv(SOURCE)
    order = [
        "B4_NO_EVIDENCE_CONTRACT",
        "B4_NO_REVISION_LOOP",
        "B4_NO_VERIFIER",
        "B4_NO_PERSISTENT_STRUCTURED_STATE",
    ]
    data = data.set_index("ablation").loc[order].reset_index()
    y = np.arange(len(data))[::-1]
    panel_heading(
        ax,
        letter,
        "Closed-loop component ablations",
        "Patient-clustered mean difference versus full B4",
        label_x=-0.11,
    )
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0, color=MID, linewidth=0.8, linestyle="--", zorder=1)

    for index, row in data.iterrows():
        estimate = row["difference"] * 100
        low = row["ci_low"] * 100
        high = row["ci_high"] * 100
        color = B4_LIGHT if index == 0 else B4_COLOR
        ax.errorbar(
            estimate,
            y[index],
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            markersize=6.2,
            markerfacecolor=color,
            markeredgecolor=B4_COLOR,
            markeredgewidth=0.8,
            ecolor=B4_COLOR,
            elinewidth=1.3,
            capsize=2.8,
            zorder=3,
        )
        text_x = high + 2 if index > 0 else -14
        ax.text(
            text_x,
            y[index],
            f"{estimate:.1f} pp\n{p_label(row['p_holm'])}",
            va="center",
            fontsize=5.25,
            color=TEXT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data["display_name"])
    ax.set_xlim(-103, 16)
    ax.set_xlabel("Ablation minus full B4 (percentage points)")


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    draw(ax)
    fig.subplots_adjust(left=0.35, right=0.98, top=0.89, bottom=0.22)
    save_publication_figure(fig, Path(__file__).with_name("panel_c_ablation_effects"))
    plt.close(fig)


if __name__ == "__main__":
    main()
