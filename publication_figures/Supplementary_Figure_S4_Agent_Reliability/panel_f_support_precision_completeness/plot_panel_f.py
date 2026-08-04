from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supp_s4_style import SYSTEM_COLORS, MID, panel_heading, quiet_axis, save_publication_figure


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "f") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    rng = np.random.default_rng(20260729)
    for system, subset in data.groupby("system", sort=False):
        x = subset["supported_claim_precision"].to_numpy(float)
        y = subset["citation_completeness"].to_numpy(float)
        xj = np.clip(x + rng.normal(0, 0.012, len(x)), -0.03, 1.03)
        yj = np.clip(y + rng.normal(0, 0.012, len(y)), -0.03, 1.03)
        label = "B2 single controller" if system.startswith("B2") else "B4 closed loop"
        ax.scatter(xj, yj, s=10, alpha=0.30, color=SYSTEM_COLORS[system],
                   edgecolor="none", label=label)
        ax.scatter(x.mean(), y.mean(), s=48, marker="D", color=SYSTEM_COLORS[system],
                   edgecolor="white", linewidth=0.8, zorder=5)
    panel_heading(ax, letter, "Reference-based support and citation completeness",
                  "Run-level results against frozen exact-string and assigned-passage references")
    ax.set_xlabel("Exact claim-support precision")
    ax.set_ylabel("Citation completeness")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    quiet_axis(ax, "both")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.98, "Diamonds = system means", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.1, color=MID)


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_f_support_precision_completeness")
    plt.close(fig)


if __name__ == "__main__":
    main()
