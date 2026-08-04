from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import panel_label, save_publication_figure, set_style  # noqa: E402


FAULTS = [
    "INVALID_REQUEST_FIELDS",
    "MISSING_GENE_FEATURES",
    "PERMANENT_MODEL_FAILURE",
    "MALFORMED_MODEL_OUTPUT",
    "TRANSIENT_RETRIEVAL_TIMEOUT",
    "CITATION_METADATA_MISMATCH",
    "CONFLICTING_EVIDENCE",
    "UNSUPPORTED_REQUESTED_CLAIM",
]
SYSTEMS = [
    "B2_SINGLE_LLM_WITH_TOOLS",
    "B3_MULTI_AGENT_NO_VERIFIER",
    "B4_FULL_CLOSED_LOOP",
]
SYSTEM_LABELS = ["B2", "B3", "B4"]


def heatmap(ax: plt.Axes, matrix: np.ndarray, title: str, ylabels: list[str] | None) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fault", ["#F3F4F6", "#F2C8BB", "#E07A5F"]
    )
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(SYSTEM_LABELS)))
    ax.set_xticklabels(SYSTEM_LABELS)
    ax.set_yticks(np.arange(len(FAULTS)))
    ax.set_yticklabels(ylabels if ylabels is not None else [])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=7.0, pad=4, fontweight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if value >= 0.72 else "#20242A",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw(ax: plt.Axes, letter: str | None = None) -> None:
    data = pd.read_csv(Path(__file__).resolve().parent / "source_data.csv")
    labels = (
        data.drop_duplicates("fault_type")
        .set_index("fault_type")
        .loc[FAULTS, "fault_label"]
        .tolist()
    )
    detection = np.vstack(
        [
            data[data["fault_type"].eq(fault)]
            .set_index("system")
            .loc[SYSTEMS, "failure_detection_rate"]
            .to_numpy()
            for fault in FAULTS
        ]
    )
    outcome = np.vstack(
        [
            data[data["fault_type"].eq(fault)]
            .set_index("system")
            .loc[SYSTEMS, "correct_outcome_rate"]
            .to_numpy()
            for fault in FAULTS
        ]
    )
    ax.set_axis_off()
    left = inset_axes(
        ax,
        width="47%",
        height="92%",
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    right = inset_axes(
        ax,
        width="47%",
        height="92%",
        loc="lower right",
        bbox_to_anchor=(0.0, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    heatmap(left, detection, "Failure detected (%)", labels)
    heatmap(right, outcome, "Correct recovery / safe abstention (%)", None)
    ax.text(
        0.0,
        1.035,
        "Fault-injection performance (30 cases × 3 repeats per cell)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )
    if letter:
        panel_label(ax, letter, x=-0.075, y=1.08)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    draw(ax, "f")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.08, top=0.88)
    save_publication_figure(fig, Path(__file__).resolve().parent / "panel_f")
    plt.close(fig)


if __name__ == "__main__":
    main()
