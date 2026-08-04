from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from supp_s3_style import COHORT_COLORS, GRID, MID, panel_heading, save_publication_figure


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "b") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    order = list(data.loc[data["cohort"] == "GSE14520 · GPL3921", "gene"])
    y = {gene: i for i, gene in enumerate(order[::-1])}
    offsets = {"GSE14520 · GPL3921": 0.11, "GSE116174 · GPL570": -0.11}
    for cohort, subset in data.groupby("cohort", sort=False):
        ys = [y[g] + offsets[cohort] for g in subset["gene"]]
        xs = subset["eligible_unique_probes"].to_numpy()
        ax.hlines(ys, 0, xs, color=COHORT_COLORS[cohort], alpha=0.45, linewidth=0.8)
        ax.scatter(xs, ys, s=19, color=COHORT_COLORS[cohort], edgecolor="white",
                   linewidth=0.45, zorder=3, label=cohort)
    panel_heading(ax, letter, "Prespecified gene coverage",
                  "Eligible uniquely mapped probes before median collapse")
    ax.set_yticks(range(len(order)), order[::-1])
    ax.set_xlabel("Eligible probes per gene")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4, borderaxespad=0.2)
    ax.text(0.99, 0.02, "15/15 genes on both platforms", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.4, color=MID)


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_b_probe_coverage")
    plt.close(fig)


if __name__ == "__main__":
    main()
