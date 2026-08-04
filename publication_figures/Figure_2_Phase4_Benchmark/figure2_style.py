from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SYSTEM_COLORS = {
    "B0_ENGINE_ONLY": "#B8BDC6",
    "B1_SINGLE_LLM_NO_TOOLS": "#A9BED2",
    "B2_SINGLE_LLM_WITH_TOOLS": "#4C78A8",
    "B3_MULTI_AGENT_NO_VERIFIER": "#7A6FA8",
    "B4_FULL_CLOSED_LOOP": "#E07A5F",
    "B4_NO_EVIDENCE_CONTRACT": "#E8A28F",
    "B4_NO_PERSISTENT_STRUCTURED_STATE": "#D7D9DE",
    "B4_NO_REVISION_LOOP": "#C9968A",
    "B4_NO_VERIFIER": "#B77B72",
}

SYSTEM_LABELS = {
    "B0_ENGINE_ONLY": "B0 Engine only",
    "B1_SINGLE_LLM_NO_TOOLS": "B1 LLM, no tools",
    "B2_SINGLE_LLM_WITH_TOOLS": "B2 Single agent + tools",
    "B3_MULTI_AGENT_NO_VERIFIER": "B3 Multi-agent, no verifier",
    "B4_FULL_CLOSED_LOOP": "B4 Full closed loop",
    "B4_NO_EVIDENCE_CONTRACT": "No evidence contract",
    "B4_NO_PERSISTENT_STRUCTURED_STATE": "No persistent state",
    "B4_NO_REVISION_LOOP": "No revision loop",
    "B4_NO_VERIFIER": "No verifier",
}

NEUTRAL = "#5B6573"
GRID = "#E7E9ED"
TEXT = "#20242A"
GREEN = "#3A8D5D"
RED = "#B94B4B"
RASTER_DPI = 600
# Final assembled figure: figsize=(7.25, 9.20), a full-width journal figure.


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#5E6269",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "legend.fontsize": 6.4,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, letter: str, x: float = -0.10, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def save_publication_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    png_dpi: int = 300,
    tiff_dpi: int = RASTER_DPI,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=tiff_dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
