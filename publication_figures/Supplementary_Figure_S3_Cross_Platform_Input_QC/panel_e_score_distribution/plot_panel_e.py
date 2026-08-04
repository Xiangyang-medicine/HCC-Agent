from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supp_s3_style import COHORT_COLORS, MID, panel_heading, quiet_axis, save_publication_figure


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "e") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    cohorts = list(COHORT_COLORS)
    values = [data.loc[data["cohort"] == cohort, "risk_score"].to_numpy() for cohort in cohorts]
    parts = ax.violinplot(values, positions=[0, 1], widths=0.75, showmeans=False,
                          showmedians=False, showextrema=False)
    for body, cohort in zip(parts["bodies"], cohorts):
        body.set_facecolor(COHORT_COLORS[cohort])
        body.set_edgecolor("none")
        body.set_alpha(0.24)
    rng = np.random.default_rng(2026)
    for i, (cohort, vals) in enumerate(zip(cohorts, values)):
        jitter = rng.uniform(-0.20, 0.20, len(vals))
        ax.scatter(i + jitter, vals, s=7, alpha=0.50, color=COHORT_COLORS[cohort],
                   edgecolor="white", linewidth=0.20)
        median = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        ax.vlines(i, q1, q3, color=COHORT_COLORS[cohort], linewidth=3.2, zorder=4)
        ax.scatter([i], [median], s=22, color="white", edgecolor=COHORT_COLORS[cohort],
                   linewidth=1.0, zorder=5)
        ax.text(i, 0.96, f"N={len(vals)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5.4, color=MID)
    panel_heading(ax, letter, "Frozen gene-component scores",
                  "Score distributions in the two eligible external cohorts")
    ax.set_xticks([0, 1], ["GSE14520\nGPL3921", "GSE116174\nGPL570"])
    ax.set_ylabel("Frozen M2T risk score")
    quiet_axis(ax, "y")
    ax.text(0.02, 0.02, "Distributions are not directly calibrated across platforms",
            transform=ax.transAxes, fontsize=5.0, color=MID, va="bottom")


def main() -> None:
    fig, ax = plt.subplots(figsize=(2.8, 2.5))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_e_score_distribution")
    plt.close(fig)


if __name__ == "__main__":
    main()
