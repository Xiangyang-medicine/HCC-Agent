from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import MODEL_COLORS, LIGHT, MID, TEXT, panel_heading, quiet_axis, save_publication_figure, set_style


DATA = Path(__file__).resolve().parent / "source_data.csv"


def _deterministic_jitter(n: int, width: float = 0.13) -> np.ndarray:
    if n == 1:
        return np.zeros(1)
    return np.linspace(-width, width, n)


def draw(ax: plt.Axes, letter: str = "a") -> None:
    data = pd.read_csv(DATA)
    model_order = ["M1", "M2", "M3", "M4", "M5"]
    metric_order = ["Harrell C", "Uno C"]
    base = {"Harrell C": 0, "Uno C": 6}

    ax.axhline(0.5, color=MID, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    for metric in metric_order:
        for i, model in enumerate(model_order):
            position = base[metric] + i + 1
            values = (
                data.loc[
                    data["metric"].eq(metric) & data["model_short"].eq(model), "value"
                ]
                .sort_index()
                .to_numpy()
            )
            color = MODEL_COLORS[model]
            box = ax.boxplot(
                values,
                positions=[position],
                widths=0.56,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": TEXT, "linewidth": 1.0},
                boxprops={"facecolor": color, "alpha": 0.16, "edgecolor": color, "linewidth": 0.8},
                whiskerprops={"color": color, "linewidth": 0.75},
                capprops={"color": color, "linewidth": 0.75},
                zorder=1,
            )
            _ = box
            jitter = _deterministic_jitter(len(values))
            order = np.argsort(values)
            x = np.empty_like(jitter)
            x[order] = jitter
            ax.scatter(
                position + x,
                values,
                s=8.5,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.78,
                zorder=3,
            )
            mean = float(np.mean(values))
            ax.plot(
                [position - 0.22, position + 0.22],
                [mean, mean],
                color=color,
                linewidth=1.6,
                solid_capstyle="round",
                zorder=4,
            )
            if model == "M4":
                ax.text(
                    position,
                    mean + 0.012,
                    f"{mean:.3f}",
                    color=color,
                    fontsize=5.7,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

    ax.axvline(6.0, color=LIGHT, linewidth=0.8)
    ax.text(3, 0.797, "Harrell C", ha="center", va="top", fontsize=6.4, fontweight="bold")
    ax.text(9, 0.797, "Uno C (IPCW)", ha="center", va="top", fontsize=6.4, fontweight="bold")
    ax.text(
        5.7,
        0.505,
        "chance",
        color=MID,
        fontsize=5.3,
        ha="right",
        va="bottom",
    )
    ax.set_xlim(0.35, 11.65)
    ax.set_ylim(0.24, 0.81)
    ax.set_xticks([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
    ax.set_xticklabels(model_order + model_order)
    ax.set_ylabel("Outer-fold concordance")
    panel_heading(
        ax,
        letter,
        "Discrimination across outer test folds",
        "25 folds · 5 repeats × 5 folds · thick bar = mean",
    )
    quiet_axis(ax, grid_axis="y")


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.35, 2.55))
    draw(ax)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.17)
    save_publication_figure(
        fig, Path(__file__).resolve().parent / "panel_a_model_discrimination"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
