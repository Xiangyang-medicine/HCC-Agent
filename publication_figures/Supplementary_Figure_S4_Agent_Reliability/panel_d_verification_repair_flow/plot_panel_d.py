from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

from supp_s4_style import GOLD, NAVY, PALE_NAVY, PALE_RUST, PALE_TEAL, RUST, TEAL, TEXT, panel_heading, save_publication_figure


HERE = Path(__file__).resolve().parent


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.35):
    verts = [(x0, y0a), (x0 + 0.45, y0a), (x1 - 0.45, y1a), (x1, y1a),
             (x1, y1b), (x1 - 0.45, y1b), (x0 + 0.45, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw(ax: plt.Axes, letter: str = "d") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    ax.set_xlim(-0.15, 3.58)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_heading(ax, letter, "Verification and repair flow",
                  "B4 clean runs: initial plan state to verified report or safe abstention")
    left_order = ["Valid initial plan", "Invalid initial plan"]
    right_order = ["Direct verified report", "Verified after repair", "Safe abstention", "Unresolved failure"]
    right_colors = {"Direct verified report": NAVY, "Verified after repair": TEAL,
                    "Safe abstention": GOLD, "Unresolved failure": RUST}
    right_pales = {"Direct verified report": PALE_NAVY, "Verified after repair": PALE_TEAL,
                   "Safe abstention": "#F5EDD9", "Unresolved failure": PALE_RUST}
    total = data["n_runs"].sum()
    gap = 0.025
    left_totals = data.groupby("initial_plan_state")["n_runs"].sum().reindex(left_order, fill_value=0)
    right_totals = data.groupby("terminal_state")["n_runs"].sum().reindex(right_order, fill_value=0)

    def positions(totals):
        pos, top = {}, 0.87
        for name, n in totals.items():
            height = 0 if n == 0 else (n / total) * 0.72
            pos[name] = (top - height, top)
            top -= height + gap
        return pos

    lp, rp = positions(left_totals), positions(right_totals)
    left_cursor = {k: v[0] for k, v in lp.items()}
    right_cursor = {k: v[0] for k, v in rp.items()}
    for row in data.itertuples(index=False):
        h = row.n_runs / total * 0.72
        l0, r0 = left_cursor[row.initial_plan_state], right_cursor[row.terminal_state]
        ribbon(ax, 0.65, 2.30, l0, l0 + h, r0, r0 + h, right_colors[row.terminal_state])
        left_cursor[row.initial_plan_state] += h
        right_cursor[row.terminal_state] += h

    for name, (y0, y1) in lp.items():
        n = int(left_totals[name])
        color = NAVY if name.startswith("Valid") else RUST
        ax.add_patch(Rectangle((0.15, y0), 0.5, y1 - y0, facecolor="white",
                               edgecolor=color, linewidth=0.9))
        if name.startswith("Valid"):
            ax.text(0.40, (y0 + y1) / 2, f"{name}\nN={n}", ha="center", va="center",
                    fontsize=5.2, color=color, fontweight="bold")
        else:
            ax.text(0.10, (y0 + y1) / 2, f"{name} · N={n}", ha="right", va="center",
                    fontsize=4.9, color=color, fontweight="bold")
    for name, (y0, y1) in rp.items():
        n = int(right_totals[name])
        if n == 0:
            continue
        ax.add_patch(Rectangle((2.30, y0), 0.68, y1 - y0,
                               facecolor=right_pales[name], edgecolor=right_colors[name], linewidth=0.9))
        if name == "Direct verified report":
            ax.text(2.64, (y0 + y1) / 2, f"{name}\nN={n}", ha="center", va="center",
                    fontsize=5.1, color=right_colors[name], fontweight="bold")
        else:
            ax.text(3.04, (y0 + y1) / 2, f"{name} · N={n}", ha="left", va="center",
                    fontsize=4.9, color=right_colors[name], fontweight="bold")
    ax.text(0.15, 0.94, "Initial planning", fontsize=5.4, color=TEXT, fontweight="bold")
    ax.text(2.30, 0.94, "Terminal state", fontsize=5.4, color=TEXT, fontweight="bold")


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_d_verification_repair_flow")
    plt.close(fig)


if __name__ == "__main__":
    main()
