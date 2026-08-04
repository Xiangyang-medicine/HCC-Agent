from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    data = pd.read_csv(Path(__file__).resolve().parent / "source_data.csv")
    short = {
        "B2_SINGLE_LLM_WITH_TOOLS": "B2",
        "B3_MULTI_AGENT_NO_VERIFIER": "B3",
        "B4_FULL_CLOSED_LOOP": "B4",
    }
    for _, row in data.iterrows():
        system = row["system"]
        x = row["median_latency_ms"] / 1000
        y = row["exact_three_run_agreement"] * 100
        xerr = [
            [x - row["latency_ci_low_ms"] / 1000],
            [row["latency_ci_high_ms"] / 1000 - x],
        ]
        yerr = [
            [(row["exact_three_run_agreement"] - row["agreement_ci_low"]) * 100],
            [(row["agreement_ci_high"] - row["exact_three_run_agreement"]) * 100],
        ]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="#6A6F77",
            elinewidth=0.7,
            capsize=2,
            zorder=2,
        )
        size = row["mean_total_tokens"] / 13
        ax.scatter(
            x,
            y,
            s=size,
            color=SYSTEM_COLORS[system],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        dx, dy, ha = {
            "B2_SINGLE_LLM_WITH_TOOLS": (0.30, 2.2, "left"),
            "B3_MULTI_AGENT_NO_VERIFIER": (-0.35, -5.8, "right"),
            "B4_FULL_CLOSED_LOOP": (0.26, 1.8, "left"),
        }[system]
        ax.text(
            x + dx,
            y + dy,
            f"{short[system]}  {y:.0f}%\n{x:.1f}s; {row['mean_total_tokens'] / 1000:.1f}k tok",
            fontsize=6.0,
            ha=ha,
            va="center",
        )
    ax.set_xlim(21.5, 28.8)
    ax.set_ylim(44, 100)
    ax.set_xlabel("Median wall latency (s)")
    ax.set_ylabel("Exact three-run agreement (%)")
    ax.set_title("Reliability–efficiency trade-off", loc="left", fontweight="bold", pad=5)
    clean_axis(ax, "both")
    ax.text(
        0.99,
        0.02,
        "Point area scales with mean token use",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#5B6573",
    )
    if letter:
        panel_label(ax, letter, x=-0.17, y=1.10)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    draw(ax, "e")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.82)
    save_publication_figure(fig, Path(__file__).resolve().parent / "panel_e")
    plt.close(fig)


if __name__ == "__main__":
    main()
