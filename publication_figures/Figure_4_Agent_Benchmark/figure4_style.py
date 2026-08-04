from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


B2_COLOR = "#536C8B"
B2_LIGHT = "#D9E1EA"
B4_COLOR = "#C65D3A"
B4_LIGHT = "#F2D9CF"
TEAL = "#257B78"
GOLD = "#B9862C"
RED = "#B2473E"
TEXT = "#20252B"
MID = "#66707C"
LIGHT = "#E7E9ED"
VERY_LIGHT = "#F5F6F8"
WHITE = "#FFFFFF"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.6,
            "axes.titlesize": 7.6,
            "axes.labelsize": 6.6,
            "axes.linewidth": 0.65,
            "axes.edgecolor": MID,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "legend.fontsize": 5.8,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
        }
    )


def panel_label(ax: plt.Axes, letter: str, x: float = -0.07, y: float = 1.055) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def panel_heading(
    ax: plt.Axes,
    letter: str,
    title: str,
    subtitle: str,
    *,
    label_x: float = -0.08,
    title_x: float = 0.0,
    title_y: float = 1.105,
    subtitle_y: float = 1.035,
) -> None:
    """Draw the manuscript-wide two-tier panel heading."""
    ax.text(
        label_x,
        title_y,
        letter,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )
    ax.text(
        title_x,
        title_y,
        title,
        transform=ax.transAxes,
        fontsize=7.6,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
        clip_on=False,
    )
    ax.text(
        title_x,
        subtitle_y,
        subtitle,
        transform=ax.transAxes,
        fontsize=5.35,
        ha="left",
        va="top",
        color=MID,
        clip_on=False,
    )


def quiet_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHT, linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
