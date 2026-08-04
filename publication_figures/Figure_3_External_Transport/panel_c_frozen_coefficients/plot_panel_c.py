from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure3_style import (  # noqa: E402
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


def draw(ax: plt.Axes, letter: str = "c") -> None:
    data = pd.read_csv(SOURCE).sort_values("feature_order")
    y = np.arange(len(data))[::-1]
    colors = [ORANGE if active else "#C9CED4" for active in data["nonzero"]]

    panel_heading(
        ax,
        letter,
        "Frozen gene-only coefficient profile",
        "15 prespecified metabolic genes · coefficients fixed before transport",
        label_x=-0.12,
    )
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0, color=MID, linewidth=0.75, zorder=1)
    for index, row in enumerate(data.itertuples(index=False)):
        ax.plot(
            [0, row.coefficient],
            [y[index], y[index]],
            color=colors[index],
            linewidth=1.4 if row.nonzero else 0.7,
            zorder=2,
        )
        ax.scatter(
            row.coefficient,
            y[index],
            s=22 if row.nonzero else 10,
            color=colors[index],
            edgecolor="white" if row.nonzero else colors[index],
            linewidth=0.5,
            zorder=3,
        )
        if row.nonzero:
            ax.text(
                row.coefficient + 0.006,
                y[index],
                f"{row.coefficient:.3f}",
                fontsize=5.0,
                color=ORANGE,
                va="center",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(data["gene"])
    ax.set_xlim(-0.012, 0.235)
    ax.set_xlabel("Standardized log-hazard coefficient")
    ax.text(
        0.98,
        0.99,
        f"{int(data['nonzero'].sum())}/15 non-zero",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        color=TEXT,
    )


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(2.25, 3.25))
    draw(ax)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.92, bottom=0.15)
    save_publication_figure(
        fig, Path(__file__).with_name("panel_c_frozen_coefficients")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
