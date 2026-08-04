from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from supp_s4_style import NAVY, TEAL, panel_heading, quiet_axis, save_publication_figure


HERE = Path(__file__).resolve().parent


def draw(ax: plt.Axes, letter: str = "a") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    pivot = data.pivot(index="risk_quintile", columns="event", values="n_cases").fillna(0)
    x = pivot.index.to_numpy()
    non_event = pivot.get(0, pd.Series(0, index=pivot.index)).to_numpy()
    event = pivot.get(1, pd.Series(0, index=pivot.index)).to_numpy()
    ax.bar(x, non_event, width=0.62, color=NAVY, alpha=0.86, label="No observed event")
    ax.bar(x, event, width=0.62, bottom=non_event, color=TEAL, alpha=0.90, label="Observed event")
    for xi, total in zip(x, non_event + event):
        ax.text(xi, total + 0.55, str(int(total)), ha="center", fontsize=5.4)
    panel_heading(ax, letter, "Formal benchmark composition",
                  "100 blinded cases stratified by OOF risk quintile and event status")
    ax.set_xlabel("OOF risk-quintile sampling stratum")
    ax.set_ylabel("Cases")
    ax.set_xticks(x)
    ax.set_ylim(0, max(non_event + event) * 1.18)
    quiet_axis(ax, "y")
    ax.legend(frameon=False, loc="upper right")


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_a_benchmark_composition")
    plt.close(fig)


if __name__ == "__main__":
    main()
