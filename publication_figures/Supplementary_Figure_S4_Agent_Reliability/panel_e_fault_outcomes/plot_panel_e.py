from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from supp_s4_style import GOLD, MID, RUST, TEAL, panel_heading, quiet_axis, save_publication_figure


HERE = Path(__file__).resolve().parent
B2 = "B2_SINGLE_LLM_WITH_TOOLS"
B4 = "B4_FULL_CLOSED_LOOP"
LABELS = {
    "CITATION_METADATA_MISMATCH": "Citation metadata",
    "CONFLICTING_EVIDENCE": "Conflicting evidence",
    "INVALID_REQUEST_FIELDS": "Invalid request",
    "MALFORMED_MODEL_OUTPUT": "Malformed output",
    "MISSING_GENE_FEATURES": "Missing genes",
    "PERMANENT_MODEL_FAILURE": "Permanent model failure",
    "TRANSIENT_RETRIEVAL_TIMEOUT": "Transient timeout",
    "UNSUPPORTED_REQUESTED_CLAIM": "Unsupported claim",
}


def draw(ax: plt.Axes, letter: str = "e") -> None:
    data = pd.read_csv(HERE / "source_data.csv")
    fault_order = list(LABELS)
    outcomes = ["Correct terminal outcome", "Detected, incorrect terminal", "Undetected / unrecovered"]
    colors = [TEAL, GOLD, RUST]
    y = np.arange(len(fault_order))
    height = 0.32
    for offset, system, hatch in [(-0.17, B2, ""), (0.17, B4, "///")]:
        left = np.zeros(len(fault_order))
        for outcome, color in zip(outcomes, colors):
            vals = []
            for fault in fault_order:
                match = data.loc[(data["fault_type"] == fault) & (data["system"] == system)
                                 & (data["outcome"] == outcome), "fraction"]
                vals.append(float(match.iloc[0]) if len(match) else 0.0)
            ax.barh(y + offset, vals, left=left, height=height, color=color,
                    edgecolor="white", linewidth=0.35, hatch=hatch,
                    label=outcome if system == B2 else None)
            left += np.asarray(vals)
    panel_heading(ax, letter, "Fault-injection terminal outcomes",
                  "B2 solid bars; B4 hatched bars; 90 runs per fault and system")
    ax.set_yticks(y, [LABELS[x] for x in fault_order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of fault-injected runs")
    quiet_axis(ax, "x")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    ax.text(0.99, 1.01, "B2 solid  |  B4 hatched", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.2, color=MID)


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    draw(ax)
    save_publication_figure(fig, HERE / "panel_e_fault_outcomes")
    plt.close(fig)


if __name__ == "__main__":
    main()
