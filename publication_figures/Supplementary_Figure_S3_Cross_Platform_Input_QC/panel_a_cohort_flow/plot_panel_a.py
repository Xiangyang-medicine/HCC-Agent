from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from supp_s3_style import MID, NAVY, PALE_NAVY, PALE_RUST, PALE_TEAL, RUST, TEAL, panel_heading, save_publication_figure


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "a") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    ax.set_xlim(-0.2, 3.03)
    ax.set_ylim(-0.65, 2.65)
    ax.axis("off")
    panel_heading(ax, letter, "Cohort screening and analysis scope",
                  "Two cohorts entered transport analysis; GPL571 was not analysed")
    cohorts = [
        ("GSE14520 · GPL3921", 2.0, NAVY, PALE_NAVY),
        ("GSE116174 · GPL570", 1.0, TEAL, PALE_TEAL),
        ("GSE14520 · GPL571", 0.0, RUST, PALE_RUST),
    ]
    for cohort, y, color, pale in cohorts:
        subset = data.loc[data["cohort"] == cohort].sort_values("stage_order")
        ax.text(-0.14, y, cohort, ha="right", va="center", fontsize=6.3,
                color=color, fontweight="bold")
        for i, row in enumerate(subset.itertuples(index=False)):
            x = i * 1.08
            excluded = row.decision == "excluded_insufficient_n"
            box = FancyBboxPatch(
                (x, y - 0.22), 0.86, 0.44,
                boxstyle="round,pad=0.025,rounding_size=0.04",
                facecolor=pale, edgecolor=color, linewidth=0.85,
                linestyle="--" if excluded else "-",
            )
            ax.add_patch(box)
            if excluded:
                label = f"{row.stage}\nN = 21\nExcluded · insufficient N"
            elif i == 2:
                label = f"{row.stage}\nN = {row.n}\nIncluded"
            else:
                label = f"{row.stage}\nN = {row.n}"
            ax.text(x + 0.43, y, label, ha="center", va="center",
                    fontsize=5.25, color=color, linespacing=1.12,
                    fontweight="bold" if i == 2 else "normal")
            if i < 2:
                ax.add_patch(FancyArrowPatch(
                    (x + 0.88, y), (x + 1.04, y), arrowstyle="-|>",
                    mutation_scale=7, linewidth=0.75, color=MID,
                ))
    ax.text(0.0, -0.50, "OS, overall survival. Mapping and standardisation were outcome-blind.",
            fontsize=5.1, color=MID, ha="left")


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_a_cohort_flow")
    plt.close(fig)


if __name__ == "__main__":
    main()
