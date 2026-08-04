from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supp_s4_style import NAVY, TEAL, MID, panel_heading, quiet_axis, save_publication_figure


HERE = Path(__file__).resolve().parent
B2 = "B2_SINGLE_LLM_WITH_TOOLS"
B4 = "B4_FULL_CLOSED_LOOP"


def draw(ax: plt.Axes, letter: str = "b") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    rng = np.random.default_rng(20260729)
    for row in data.itertuples(index=False):
        jitter = rng.uniform(-0.035, 0.035)
        ax.plot([0 + jitter, 1 + jitter], [getattr(row, B2), getattr(row, B4)],
                color=TEAL if row.delta_b4_minus_b2 > 0 else MID,
                alpha=0.20, linewidth=0.65, zorder=1)
    ax.scatter(np.full(len(data), 0) + rng.uniform(-0.045, 0.045, len(data)), data[B2],
               s=9, color=NAVY, alpha=0.55, edgecolor="white", linewidth=0.25, zorder=2)
    ax.scatter(np.full(len(data), 1) + rng.uniform(-0.045, 0.045, len(data)), data[B4],
               s=9, color=TEAL, alpha=0.55, edgecolor="white", linewidth=0.25, zorder=2)
    ax.scatter([0, 1], [data[B2].mean(), data[B4].mean()], marker="D", s=34,
               color=[NAVY, TEAL], edgecolor="white", linewidth=0.7, zorder=5)
    panel_heading(ax, letter, "Paired case-level task success",
                  "Each line links the same case; fraction successful across three repeats")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 1], ["B2 single\ncontroller", "B4 closed\nloop"])
    ax.set_ylabel("Case-level pass fraction")
    quiet_axis(ax, "y")
    improved = int((data["delta_b4_minus_b2"] > 0).sum())
    worsened = int((data["delta_b4_minus_b2"] < 0).sum())
    ax.text(0.98, 0.03, f"Improved {improved} · worsened {worsened}",
            transform=ax.transAxes, ha="right", fontsize=5.2, color=MID)


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_b_paired_case_success")
    plt.close(fig)


if __name__ == "__main__":
    main()
