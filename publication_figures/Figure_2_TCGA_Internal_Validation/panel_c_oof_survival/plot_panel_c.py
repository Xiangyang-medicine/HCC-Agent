from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import LIGHT, MID, NAVY, RUST, panel_heading, quiet_axis, save_publication_figure, set_style


HERE = Path(__file__).resolve().parent


def _p_text(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def draw(ax: plt.Axes, letter: str = "c") -> None:
    curves = pd.read_csv(HERE / "source_data_km_curves.csv")
    at_risk = pd.read_csv(HERE / "source_data_at_risk.csv")
    stats = json.loads((HERE / "statistics.json").read_text(encoding="utf-8"))
    colors = {"Low": NAVY, "High": RUST}

    for group in ["Low", "High"]:
        subset = curves.loc[curves["risk_group"].eq(group)]
        ax.step(
            subset["timeline_months"],
            subset["survival_probability"],
            where="post",
            color=colors[group],
            linewidth=1.5,
            label=f"{group} OOF risk",
            zorder=3,
        )
        ax.fill_between(
            subset["timeline_months"],
            subset["ci_lower"],
            subset["ci_upper"],
            step="post",
            color=colors[group],
            alpha=0.11,
            linewidth=0,
            zorder=2,
        )

    ax.set_xlim(0, 96)
    ax.set_ylim(0.18, 1.02)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Overall survival (months)", labelpad=2)
    ax.set_ylabel("Survival probability")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), ncol=1)
    ax.text(
        0.98,
        0.97,
        (
            f"HR = {stats['cox_hr_high_vs_low']:.2f} "
            f"({stats['cox_hr_ci_lower']:.2f}–{stats['cox_hr_ci_upper']:.2f})\n"
            f"log-rank P = {_p_text(stats['logrank_p_value'])}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        linespacing=1.35,
    )

    ax.text(
        -0.03,
        -0.225,
        "No. at risk",
        transform=ax.transAxes,
        fontsize=5.4,
        fontweight="bold",
        ha="right",
        va="center",
        clip_on=False,
    )
    for row_index, group in enumerate(["Low", "High"]):
        y = -0.29 - row_index * 0.075
        ax.text(
            -0.03,
            y,
            group,
            transform=ax.transAxes,
            fontsize=5.3,
            color=colors[group],
            ha="right",
            va="center",
            clip_on=False,
        )
        subset = at_risk.loc[at_risk["risk_group"].eq(group)]
        for _, row in subset.iterrows():
            x = float(row["time_months"]) / 96.0
            ax.text(
                x,
                y,
                str(int(row["n_at_risk"])),
                transform=ax.transAxes,
                fontsize=5.2,
                color=colors[group],
                ha="center",
                va="center",
                clip_on=False,
            )
    panel_heading(
        ax,
        letter,
        "OOF risk stratifies overall survival",
        "Fold-wise risk ranks averaged across five OOF predictions · median split",
    )
    quiet_axis(ax, grid_axis="y")


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.25, 2.95))
    draw(ax)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.28)
    save_publication_figure(fig, HERE / "panel_c_oof_survival")
    plt.close(fig)


if __name__ == "__main__":
    main()
