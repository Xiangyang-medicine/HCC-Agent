from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supp_s2_style import RUST, TEXT, panel_heading, set_style  # noqa: E402


SOURCE = Path(__file__).with_name("source_data.csv")


def draw(ax: plt.Axes, letter: str = "d") -> None:
    data = pd.read_csv(SOURCE)
    panel_heading(
        ax,
        letter,
        "M5 repeat-pair risk density",
        "Same display as panel c; diffuse density indicates instability",
    )
    cmap = LinearSegmentedColormap.from_list("m5_transition", ["#FFFFFF", "#F0D5CA", RUST])
    ax.hexbin(data["risk_percentile_a"], data["risk_percentile_b"], gridsize=27, extent=(0, 1, 0, 1), cmap=cmap, mincnt=1, bins="log", linewidths=0)
    ax.plot([0, 1], [0, 1], color="#667382", linestyle=(0, (3, 2)), linewidth=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Risk percentile in repeat A")
    ax.set_ylabel("Risk percentile in repeat B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 6.6, "svg.fonttype": "none", "pdf.fonttype": 42})
    set_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    draw(ax)
    fig.subplots_adjust(left=0.18, right=0.92, top=0.81, bottom=0.20)
    stem = Path(__file__).with_name("panel_d_m5_quintile_transition")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
