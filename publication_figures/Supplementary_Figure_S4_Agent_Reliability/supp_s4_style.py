from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


TEXT = "#20262E"
MID = "#667382"
GRID = "#DDE3E8"
NAVY = "#315B7D"
TEAL = "#168F78"
RUST = "#C75A34"
GOLD = "#C49432"
SLATE = "#89949F"
PALE_NAVY = "#E7EEF4"
PALE_TEAL = "#E2F2EE"
PALE_RUST = "#F6E6DF"
SYSTEM_COLORS = {
    "B2_SINGLE_LLM_WITH_TOOLS": NAVY,
    "B4_FULL_CLOSED_LOOP": TEAL,
}
SYSTEM_LABELS = {
    "B2_SINGLE_LLM_WITH_TOOLS": "B2 tool-using\nsingle controller",
    "B4_FULL_CLOSED_LOOP": "B4 verifier-guided\nclosed loop",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 8.1,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.65,
            "axes.edgecolor": MID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_heading(ax: plt.Axes, letter: str, title: str, subtitle: str) -> None:
    ax.text(-0.08, 1.15, letter, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", ha="right", va="top")
    ax.text(0.0, 1.15, title, transform=ax.transAxes, fontsize=8.1,
            fontweight="bold", ha="left", va="top")
    ax.text(0.0, 1.05, subtitle, transform=ax.transAxes, fontsize=5.5,
            color=MID, ha="left", va="top")


def quiet_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
