from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supp_s4_style import SYSTEM_COLORS, SYSTEM_LABELS, MID, panel_heading, quiet_axis, save_publication_figure


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "c") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    systems = list(SYSTEM_COLORS)
    rng = np.random.default_rng(42)
    for i, system in enumerate(systems):
        subset = data.loc[data["system"] == system]
        jitter = rng.uniform(-0.20, 0.20, len(subset))
        ax.scatter(i + jitter, subset["pass_fraction"], s=11, alpha=0.48,
                   color=SYSTEM_COLORS[system], edgecolor="white", linewidth=0.25)
        ax.scatter(i, subset["pass_fraction"].mean(), s=42, marker="D",
                   color=SYSTEM_COLORS[system], edgecolor="white", linewidth=0.8, zorder=4)
        agreement = subset["exact_three_run_agreement"].mean()
        ax.text(i, 0.08, f"Exact agreement\n{agreement:.0%}",
                ha="center", va="bottom", fontsize=5.25, color=MID)
    panel_heading(ax, letter, "Repeat-level reliability",
                  "Per-case success frequency and exact three-run agreement")
    ax.set_xticks(range(len(systems)), [SYSTEM_LABELS[s] for s in systems])
    ax.set_ylabel("Success fraction across repeats")
    ax.set_ylim(-0.04, 1.05)
    quiet_axis(ax, "y")


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_c_repeat_reliability")
    plt.close(fig)


if __name__ == "__main__":
    main()
