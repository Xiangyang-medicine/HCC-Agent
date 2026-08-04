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


def draw(ax: plt.Axes, letter: str | None = None) -> None:
    folder = Path(__file__).resolve().parent
    data = pd.read_csv(folder / "source_data.csv").iloc[::-1].reset_index(drop=True)
    comparison = pd.read_csv(folder / "primary_comparison.csv").iloc[0]
    y = np.arange(len(data))
    values = data["rate"].to_numpy() * 100
    low = data["ci_low"].to_numpy() * 100
    high = data["ci_high"].to_numpy() * 100
    colors = [SYSTEM_COLORS[value] for value in data["system"]]

    ax.barh(y, values, height=0.62, color=colors, edgecolor="white", linewidth=0.6, zorder=2)
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - low, high - values]),
        fmt="none",
        ecolor="#2E333A",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        zorder=3,
    )
    for yi, value, hi, success in zip(y, values, high, data["successes"]):
        ax.text(
            max(value + 1.1, hi + 1.1),
            yi,
            f"{value:.1f}% ({int(success)}/300)",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold" if value == values.max() else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(data["display_name"])
    ax.set_xlim(0, 112)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Verified task success (%)")
    ax.set_title(
        "Prespecified primary endpoint on clean cases",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    clean_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    p_value = float(comparison["p_value"])
    p_text = f"{p_value:.1e}" if p_value >= 1e-4 else f"{p_value:.1e}"
    ax.text(
        0.995,
        0.965,
        (
            f"B4 − B2: +{comparison['difference'] * 100:.1f} pp "
            f"(95% CI {comparison['ci_low'] * 100:.1f} to "
            f"{comparison['ci_high'] * 100:.1f})\n"
            f"paired permutation p={p_text}; 100 cases × 3 repeats"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF7F3",
            "edgecolor": "#E6B7A8",
            "linewidth": 0.6,
        },
    )
    if letter:
        panel_label(ax, letter, x=-0.075, y=1.08)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.1, 2.25))
    draw(ax, "a")
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.86)
    save_publication_figure(fig, Path(__file__).resolve().parent / "panel_a")
    plt.close(fig)


if __name__ == "__main__":
    main()
