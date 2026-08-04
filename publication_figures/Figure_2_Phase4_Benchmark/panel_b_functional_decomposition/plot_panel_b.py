from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_style import panel_label, save_publication_figure, set_style  # noqa: E402


SYSTEMS = [
    "B1_SINGLE_LLM_NO_TOOLS",
    "B2_SINGLE_LLM_WITH_TOOLS",
    "B3_MULTI_AGENT_NO_VERIFIER",
    "B4_FULL_CLOSED_LOOP",
]
METRICS = [
    "plan_valid",
    "tool_selection_f1",
    "schema_valid",
    "numeric_fidelity",
    "external_verifier_passed",
]


def draw(ax: plt.Axes, letter: str | None = None) -> None:
    data = pd.read_csv(Path(__file__).resolve().parent / "source_data.csv")
    value = data.pivot(index="display_name", columns="metric", values="value")
    row_labels = [
        "B1 No tools",
        "B2 Single + tools",
        "B3 Multi, no verifier",
        "B4 Closed loop",
    ]
    matrix = np.vstack(
        [
            data[data["system"].eq(system)]
            .set_index("metric")
            .loc[METRICS, "value"]
            .to_numpy()
            for system in SYSTEMS
        ]
    )
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "functional", ["#F4F5F7", "#BED0E2", "#4C78A8"]
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    labels = [
        "Plan\nvalid",
        "Tool\nF1",
        "Schema\nvalid",
        "Numeric\nfidelity",
        "Verifier\npassed",
    ]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            val = matrix[row, col]
            ax.text(
                col,
                row,
                f"{val * 100:.0f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if val >= 0.72 else "#20242A",
                fontweight="bold" if SYSTEMS[row] == "B4_FULL_CLOSED_LOOP" else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Functional endpoint decomposition (%)", loc="left", fontweight="bold", pad=5)
    if letter:
        panel_label(ax, letter, x=-0.20, y=1.10)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    draw(ax, "b")
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.18, top=0.82)
    save_publication_figure(fig, Path(__file__).resolve().parent / "panel_b")
    plt.close(fig)


if __name__ == "__main__":
    main()
