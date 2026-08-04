from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import MID, TEAL, TEXT, panel_heading, save_publication_figure, set_style


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "e") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    models = ["M1", "M2", "M3", "M4", "M5"]
    analyses = ["SA1", "SA2", "SA3"]
    metrics = ["Harrell C", "Uno C", "IBS"]
    columns = [(analysis, metric) for analysis in analyses for metric in metrics]

    values = np.zeros((len(models), len(columns)))
    ranks = np.zeros_like(values)
    for row_index, model in enumerate(models):
        for column_index, (analysis, metric) in enumerate(columns):
            row = data.loc[
                data["model_short"].eq(model)
                & data["analysis"].eq(analysis)
                & data["metric"].eq(metric)
            ].iloc[0]
            values[row_index, column_index] = float(row["value"])
            ranks[row_index, column_index] = int(row["rank"])

    cmap = mpl.colors.ListedColormap(
        ["#D6EEE7", "#E5E9ED", "#F0EEE7", "#EBD9CF", "#D9B8AA"]
    )
    norm = mpl.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    image = ax.imshow(ranks, cmap=cmap, norm=norm, aspect="auto")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=5.4,
                fontweight="bold" if models[i] == "M4" else "normal",
                color=TEXT,
            )
    ax.add_patch(
        Rectangle(
            (-0.49, models.index("M4") - 0.49),
            len(columns) - 0.02,
            0.98,
            fill=False,
            edgecolor=TEAL,
            linewidth=1.25,
            clip_on=False,
        )
    )
    for separator in [2.5, 5.5]:
        ax.axvline(separator, color="white", linewidth=2.0)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        [
            "Harrell C",
            "Uno C",
            "IBS",
            "Harrell C",
            "Uno C",
            "IBS",
            "Harrell C",
            "Uno C",
            "IBS",
        ],
        rotation=0,
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(
        [
            "M1  Clinical Cox",
            "M2  Gene elastic-net",
            "M3  Comb. elastic-net",
            "M4  Combined RSF",
            "M5  DeepSurv",
        ]
    )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.text(1, -0.98, "SA1 · primary (N=363)", ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    ax.text(4, -0.98, "SA2 · age <18 excluded (N=361)", ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    ax.text(7, -0.98, "SA3 · complete covariates (N=338)", ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    panel_heading(
        ax,
        letter,
        "Sensitivity to cohort definition",
        "Cell = estimate · shading = within-analysis rank (1 = best)",
        label_x=-0.11,
        title_y=1.58,
        subtitle_y=1.43,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        fraction=0.018,
        pad=0.015,
        ticks=[1, 2, 3, 4, 5],
    )
    cbar.ax.set_ylabel("Rank (1 = best)", rotation=90, labelpad=4)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=5.2, width=0.45, length=1.7)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.05, 2.35))
    draw(ax)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.70, bottom=0.08)
    save_publication_figure(fig, HERE / "panel_e_sensitivity")
    plt.close(fig)


if __name__ == "__main__":
    main()
