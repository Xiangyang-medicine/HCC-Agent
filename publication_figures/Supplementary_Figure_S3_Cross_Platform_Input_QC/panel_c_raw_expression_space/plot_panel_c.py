from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from supp_s3_style import COHORT_COLORS, MID, panel_heading, quiet_axis, save_publication_figure


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "c") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    for cohort, subset in data.groupby("cohort", sort=False):
        ax.scatter(subset["pc1"], subset["pc2"], s=11, alpha=0.62,
                   color=COHORT_COLORS[cohort], edgecolor="white", linewidth=0.25,
                   label=f"{cohort} (N={len(subset)})", rasterized=False)
        ax.scatter(subset["pc1"].mean(), subset["pc2"].mean(), s=45,
                   marker="X", color=COHORT_COLORS[cohort], edgecolor="white", linewidth=0.7)
    v1 = data["pc1_variance_percent"].iloc[0]
    v2 = data["pc2_variance_percent"].iloc[0]
    panel_heading(ax, letter, "Raw 15-gene expression space",
                  "PCA after probe collapse, before cohort-wise standardisation")
    ax.set_xlabel(f"PC1 ({v1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({v2:.1f}% variance)")
    quiet_axis(ax, "both")
    ax.legend(frameon=False, loc="best", handletextpad=0.35)
    ax.text(0.02, 0.02, "× cohort centroid", transform=ax.transAxes,
            fontsize=5.2, color=MID, va="bottom")


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_c_raw_expression_space")
    plt.close(fig)


if __name__ == "__main__":
    main()
