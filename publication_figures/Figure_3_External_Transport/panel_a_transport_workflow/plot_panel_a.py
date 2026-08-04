from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure3_style import (  # noqa: E402
    BLUE,
    MID,
    ORANGE,
    TEXT,
    panel_heading,
    save_publication_figure,
    set_style,
)


SOURCE = Path(__file__).with_name("source_data.csv")
GROUPS = [("Higher risk", ORANGE), ("Lower risk", BLUE)]
RISK_TIMES = [0, 24, 48]


def at_risk_counts(data: pd.DataFrame, group: str) -> list[int]:
    subset = data[data["risk_group"].eq(group)]
    return [
        int((subset["survival_months"] >= time).sum())
        for time in RISK_TIMES
    ]


def draw_cohort(axis: plt.Axes, data: pd.DataFrame, title: str) -> None:
    high = data[data["risk_group"].eq("Higher risk")]
    low = data[data["risk_group"].eq("Lower risk")]
    logrank = logrank_test(
        high["survival_months"],
        low["survival_months"],
        event_observed_A=high["event"],
        event_observed_B=low["event"],
    )
    cox_data = data[["survival_months", "event"]].copy()
    cox_data["higher_risk"] = data["risk_group"].eq("Higher risk").astype(int)
    cox = CoxPHFitter()
    cox.fit(
        cox_data,
        duration_col="survival_months",
        event_col="event",
        formula="higher_risk",
    )
    cox_summary = cox.summary.loc["higher_risk"]

    for group, color in GROUPS:
        subset = data[data["risk_group"].eq(group)]
        km = KaplanMeierFitter(label=f"{group} (n={len(subset)})")
        km.fit(
            subset["survival_months"],
            event_observed=subset["event"],
        )
        km.plot_survival_function(
            ax=axis,
            ci_show=True,
            ci_alpha=0.10,
            show_censors=True,
            censor_styles={
                "marker": "|",
                "ms": 3.2,
                "mew": 0.7,
            },
            color=color,
            linewidth=1.25,
        )

    axis.set_title(title, fontsize=6.2, fontweight="bold", pad=2)
    axis.set_xlim(0, 70)
    axis.set_ylim(0, 1.03)
    axis.set_xticks([0, 24, 48, 72])
    axis.set_xticklabels(["0", "24", "48", "72"])
    axis.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    axis.set_xlabel("")
    axis.set_ylabel("Overall survival")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    axis.legend(
        loc="lower left",
        frameon=False,
        fontsize=4.7,
        handlelength=1.4,
        borderaxespad=0.2,
    )
    p_text = (
        "P<0.001"
        if logrank.p_value < 0.001
        else f"P={logrank.p_value:.3f}"
    )
    axis.text(
        0.98,
        0.97,
        (
            f"HR {cox_summary['exp(coef)']:.2f} "
            f"[{cox_summary['exp(coef) lower 95%']:.2f}–"
            f"{cox_summary['exp(coef) upper 95%']:.2f}]\n"
            f"Log-rank {p_text}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=TEXT,
    )

    axis.text(
        0.0,
        -0.24,
        "No. at risk",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=5.0,
        color=MID,
    )
    for row_index, (group, color) in enumerate(GROUPS):
        counts = at_risk_counts(data, group)
        axis.text(
            0.0,
            -0.34 - row_index * 0.10,
            "High" if group == "Higher risk" else "Low",
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=5.0,
            color=color,
            fontweight="bold",
        )
        for x, count in zip([0.24, 0.57, 0.90], counts):
            axis.text(
                x,
                -0.34 - row_index * 0.10,
                str(count),
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=5.0,
                color=TEXT,
            )
    for x, time in zip([0.24, 0.57, 0.90], RISK_TIMES):
        axis.text(
            x,
            -0.24,
            str(time),
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=5.0,
            color=MID,
        )


def draw(ax: plt.Axes, letter: str = "a") -> None:
    data = pd.read_csv(SOURCE)
    ax.set_axis_off()
    panel_heading(
        ax,
        letter,
        "Frozen gene-only score stratifies external survival",
        "Locked TCGA median cutoff · no external outcome-driven threshold",
        label_x=-0.08,
    )
    left = ax.inset_axes([0.00, 0.31, 0.45, 0.57])
    right = ax.inset_axes([0.55, 0.31, 0.45, 0.57])
    draw_cohort(
        left,
        data[
            data["cohort"].eq("GSE14520")
            & data["platform"].eq("GPL3921")
        ],
        "GSE14520 / GPL3921",
    )
    draw_cohort(
        right,
        data[
            data["cohort"].eq("GSE116174")
            & data["platform"].eq("GPL570")
        ],
        "GSE116174 / GPL570",
    )


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.1, 2.7))
    draw(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.08)
    save_publication_figure(
        fig, Path(__file__).with_name("panel_a_transport_workflow")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
