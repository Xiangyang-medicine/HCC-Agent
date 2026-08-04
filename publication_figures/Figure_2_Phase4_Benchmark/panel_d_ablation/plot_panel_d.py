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


ORDER = [
    "B4_FULL_CLOSED_LOOP",
    "B4_NO_EVIDENCE_CONTRACT",
    "B4_NO_PERSISTENT_STRUCTURED_STATE",
    "B4_NO_REVISION_LOOP",
    "B4_NO_VERIFIER",
]


def p_label(value: float) -> str:
    if value < 0.001:
        return "pHolm<0.001"
    return f"pHolm={value:.3f}"


def draw(ax: plt.Axes, letter: str | None = None) -> None:
    folder = Path(__file__).resolve().parent
    data = pd.read_csv(folder / "source_data.csv").set_index("system").loc[ORDER].reset_index()
    comparisons = pd.read_csv(folder / "paired_comparisons.csv").set_index("ablation")
    y = np.arange(len(data))
    values = data["rate"].to_numpy() * 100
    low = data["ci_low"].to_numpy() * 100
    high = data["ci_high"].to_numpy() * 100
    colors = [SYSTEM_COLORS[value] for value in data["system"]]

    ax.barh(y, values, height=0.62, color=colors, zorder=2)
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - low, high - values]),
        fmt="none",
        ecolor="#30343A",
        capsize=2,
        elinewidth=0.75,
        zorder=3,
    )
    for yi, row in data.iterrows():
        value = row["rate"] * 100
        text = f"{value:.1f}%"
        if row["system"] != "B4_FULL_CLOSED_LOOP":
            comp = comparisons.loc[row["system"]]
            text += f"  {p_label(float(comp['p_holm']))}"
        ax.text(max(value + 1.2, row["ci_high"] * 100 + 1.2), yi, text, va="center", fontsize=6.0)

    ax.set_yticks(y)
    ax.set_yticklabels(data["display_name"])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Verified task success (%)")
    ax.set_title("Closed-loop component ablations", loc="left", fontweight="bold", pad=5)
    clean_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if letter:
        panel_label(ax, letter, x=-0.20, y=1.10)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    draw(ax, "d")
    fig.subplots_adjust(left=0.37, right=0.98, bottom=0.22, top=0.82)
    save_publication_figure(fig, Path(__file__).resolve().parent / "panel_d")
    plt.close(fig)


if __name__ == "__main__":
    main()
