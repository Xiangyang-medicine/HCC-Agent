#!/usr/bin/env python3
"""Render the main-text external microarray transportability figure.

Only the two prespecified main-text strata are rendered. The small GPL571
stratum remains retained in the supplementary audit and is never deleted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = [
    ("GSE14520 GPL3921", "GSE14520_GPL3921", 221, 85),
    ("GSE116174 GPL570", "GSE116174_GPL570", 64, 27),
]
SOURCE_DIR = ROOT / "experiments" / "phase3b" / "microarray_transport"
OUTPUT_DIR = ROOT / "figures" / "publication"


def _result(cohort_name: str, label: str, n: int, events: int) -> dict:
    path = SOURCE_DIR / f"{label}_M2T_EVALUATION.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETED_SECONDARY_EXPLORATORY_CROSS_PLATFORM_EVALUATION":
        raise ValueError(f"Unexpected evaluation status for {label}.")
    if payload.get("n") != n or payload.get("events") != events:
        raise ValueError(f"Cohort size/event mismatch for {label}.")
    uno = payload["bootstrap"]["metrics"]["uno_c"]
    return {"cohort": cohort_name, "n": n, "events": events, "uno_c": uno["point_estimate"], "ci_low": uno["ci95"][0], "ci_high": uno["ci95"][1]}


def main() -> int:
    rows = [_result(name, label, n, events) for name, label, n, events in RESULTS]
    data = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "figure_external_microarray_transport_data.csv", index=False)

    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.1, 2.6), constrained_layout=True)
    y = list(range(len(data)))[::-1]
    for position, row in zip(y, data.itertuples(index=False)):
        ax.errorbar(
            row.uno_c,
            position,
            xerr=[[row.uno_c - row.ci_low], [row.ci_high - row.uno_c]],
            fmt="o",
            color="#0072B2",
            ecolor="#0072B2",
            markersize=6,
            capsize=3,
            elinewidth=1.5,
            zorder=3,
        )
        ax.text(0.705, position, f"{row.uno_c:.3f} ({row.ci_low:.3f}–{row.ci_high:.3f})", va="center", ha="left", fontsize=8)
    ax.axvline(0.5, color="#6E6E6E", linewidth=1, linestyle="--", zorder=1)
    ax.set_xlim(0.40, 0.85)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Uno C-index (95% patient-bootstrap CI)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.cohort}\nN={row.n}, events={row.events}" for row in data.itertuples(index=False)])
    ax.set_title("External cross-platform evaluation of the frozen 15-gene transport model", loc="left", fontsize=10, fontweight="bold")
    ax.text(0.5, -0.30, "Dashed line indicates chance discrimination. Results are secondary/exploratory; not an external validation of M4.", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUTPUT_DIR / "figure_external_microarray_transport.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_external_microarray_transport.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
