from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_v2_style import (  # noqa: E402
    B4_COLOR,
    LIGHT,
    MID,
    TEAL,
    panel_label,
    quiet_axis,
    save_publication_figure,
    set_style,
)


SOURCE = Path(__file__).with_name("source_data.csv")


def draw(ax: plt.Axes, letter: str = "e") -> None:
    data = pd.read_csv(SOURCE)
    order = [
        "INVALID_REQUEST_FIELDS",
        "MISSING_GENE_FEATURES",
        "PERMANENT_MODEL_FAILURE",
        "MALFORMED_MODEL_OUTPUT",
        "TRANSIENT_RETRIEVAL_TIMEOUT",
        "CITATION_METADATA_MISMATCH",
        "CONFLICTING_EVIDENCE",
        "UNSUPPORTED_REQUESTED_CLAIM",
    ]
    labels = (
        data.drop_duplicates("fault_type")
        .set_index("fault_type")
        .loc[order]["fault_label"]
        .tolist()
    )
    y = np.arange(len(order))[::-1]
    panel_label(ax, letter, x=-0.05)
    ax.set_title(
        "Fault-handling advantage of B4",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    quiet_axis(ax, grid_axis="x")
    ax.axvline(0, color=MID, linewidth=0.8, linestyle="--", zorder=1)

    specs = [
        ("failure_detected", "Failure detection", B4_COLOR, "o", 0.12),
        ("correct_terminal_outcome", "Correct terminal outcome", TEAL, "s", -0.12),
    ]
    for metric, label, color, marker, offset in specs:
        subset = data[data["metric"].eq(metric)].set_index("fault_type").loc[order]
        for index, row in enumerate(subset.itertuples()):
            estimate = row.difference_b4_minus_b2 * 100
            low = row.ci_low * 100
            high = row.ci_high * 100
            ax.errorbar(
                estimate,
                y[index] + offset,
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker,
                markersize=4.8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.1,
                label=label if index == 0 else None,
                zorder=3,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-6, 106)
    ax.set_ylim(-0.82, 7.55)
    ax.set_xlabel("B4 minus B2 (percentage points)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.63, 1.05),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    ax.axhspan(-0.42, 0.42, facecolor=LIGHT, alpha=0.55, zorder=0)
    ax.text(
        -5.8,
        -0.68,
        (
            "* B4 safely exited; the frozen terminal rule scored that exit as unsuccessful."
        ),
        fontsize=5.0,
        color=MID,
    )


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    draw(ax)
    fig.subplots_adjust(left=0.25, right=0.99, top=0.88, bottom=0.24)
    save_publication_figure(fig, Path(__file__).with_name("panel_e_fault_handling"))
    plt.close(fig)


if __name__ == "__main__":
    main()
