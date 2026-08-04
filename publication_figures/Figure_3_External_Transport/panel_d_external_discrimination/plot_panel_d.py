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
    LIGHT,
    MID,
    ORANGE,
    TEXT,
    panel_heading,
    quiet_axis,
    save_publication_figure,
    set_style,
)


SOURCE = Path(__file__).with_name("source_data.csv")


def draw(ax: plt.Axes, letter: str = "d") -> None:
    data = pd.read_csv(SOURCE)
    order = [
        ("GSE14520", "harrell_c"),
        ("GSE14520", "uno_c"),
        ("GSE116174", "harrell_c"),
        ("GSE116174", "uno_c"),
    ]
    data = (
        data.set_index(["cohort", "metric"])
        .loc[order]
        .reset_index()
    )
    y = [3.2, 2.6, 1.45, 0.85]
    colors = [ORANGE, ORANGE, BLUE, BLUE]
    markers = ["o", "s", "o", "s"]
    labels = [
        "GSE14520 / GPL3921 · Harrell C",
        "GSE14520 / GPL3921 · Uno C",
        "GSE116174 / GPL570 · Harrell C",
        "GSE116174 / GPL570 · Uno C",
    ]

    panel_heading(
        ax,
        letter,
        "External discrimination in two cohorts",
        "Harrell and Uno C · 1,000 patient-bootstrap resamples · cohorts not pooled",
        label_x=-0.08,
    )
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0.5, color=MID, linewidth=0.8, linestyle="--", zorder=1)
    ax.axhspan(2.25, 3.55, color="#FAF2EF", zorder=0)
    ax.axhspan(0.50, 1.80, color="#F0F4F8", zorder=0)

    for index, row in enumerate(data.itertuples(index=False)):
        ax.errorbar(
            row.estimate,
            y[index],
            xerr=[
                [row.estimate - row.ci_low],
                [row.ci_high - row.estimate],
            ],
            fmt=markers[index],
            markersize=5.4,
            markerfacecolor=colors[index],
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=colors[index],
            elinewidth=1.2,
            capsize=2.5,
            zorder=3,
        )
        ax.text(
            0.815,
            y[index],
            f"{row.estimate:.3f}  [{row.ci_low:.3f}–{row.ci_high:.3f}]",
            fontsize=5.0,
            color=TEXT,
            ha="right",
            va="center",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.43, 0.82)
    ax.set_ylim(0.35, 3.65)
    ax.set_xlabel("Concordance index (95% patient-bootstrap CI)")
    ax.text(
        0.505,
        3.55,
        "chance",
        fontsize=5.0,
        color=MID,
        va="top",
    )
    ax.text(
        0.99,
        0.02,
        "1,000 valid bootstrap iterations per estimate; cohorts not pooled",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    draw(ax)
    fig.subplots_adjust(left=0.40, right=0.98, top=0.92, bottom=0.16)
    save_publication_figure(
        fig, Path(__file__).with_name("panel_d_external_discrimination")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
