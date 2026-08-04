from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure2_v2_style import (  # noqa: E402
    B2_COLOR,
    B2_LIGHT,
    B4_COLOR,
    B4_LIGHT,
    MID,
    TEXT,
    panel_label,
    save_publication_figure,
    set_style,
)


def node(ax, xy, width, height, text, face, edge, *, fontsize=5.15):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=0.75,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        transform=ax.transAxes,
    )


def arrow(ax, start, end, color=MID, *, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color=color,
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def draw(ax: plt.Axes, letter: str = "a") -> None:
    ax.set_axis_off()
    panel_label(ax, letter, x=-0.03, y=1.035)
    ax.set_title(
        "System architectures",
        loc="left",
        fontweight="bold",
        pad=3,
    )

    # B2 row.
    ax.add_patch(
        FancyBboxPatch(
            (0.00, 0.57),
            1.00,
            0.31,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0,
            facecolor=B2_LIGHT,
            alpha=0.55,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.02,
        0.83,
        "B2  Tool-using single controller",
        color=B2_COLOR,
        fontweight="bold",
        fontsize=6.2,
        transform=ax.transAxes,
    )
    node(ax, (0.03, 0.63), 0.17, 0.12, "Request", "white", B2_COLOR)
    node(ax, (0.27, 0.63), 0.21, 0.12, "LLM controller", "white", B2_COLOR)
    node(ax, (0.55, 0.63), 0.18, 0.12, "Frozen tools", "white", B2_COLOR)
    node(ax, (0.80, 0.63), 0.16, 0.12, "Report", "white", B2_COLOR)
    arrow(ax, (0.20, 0.69), (0.27, 0.69), B2_COLOR)
    arrow(ax, (0.48, 0.69), (0.55, 0.69), B2_COLOR)
    arrow(ax, (0.73, 0.69), (0.80, 0.69), B2_COLOR)
    ax.text(
        0.79,
        0.585,
        "no internal verifier",
        fontsize=4.9,
        color=MID,
        transform=ax.transAxes,
    )

    # B4 row.
    ax.add_patch(
        FancyBboxPatch(
            (0.00, 0.08),
            1.00,
            0.39,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0,
            facecolor=B4_LIGHT,
            alpha=0.55,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.02,
        0.41,
        "B4  Verifier-guided closed loop",
        color=B4_COLOR,
        fontweight="bold",
        fontsize=6.2,
        transform=ax.transAxes,
    )
    positions = [
        (0.01, 0.19, 0.13, "Request"),
        (0.19, 0.19, 0.14, "Planner"),
        (0.38, 0.19, 0.16, "Frozen tools"),
        (0.59, 0.19, 0.16, "Synthesis"),
        (0.80, 0.19, 0.17, "Verifier"),
    ]
    for x, y, w, text in positions:
        node(ax, (x, y), w, 0.12, text, "white", B4_COLOR)
    for start, end in [
        ((0.14, 0.25), (0.19, 0.25)),
        ((0.33, 0.25), (0.38, 0.25)),
        ((0.54, 0.25), (0.59, 0.25)),
        ((0.75, 0.25), (0.80, 0.25)),
    ]:
        arrow(ax, start, end, B4_COLOR)
    arrow(
        ax,
        (0.88, 0.19),
        (0.26, 0.19),
        B4_COLOR,
        connectionstyle="arc3,rad=-0.32",
    )
    ax.text(
        0.57,
        0.09,
        "replan / retry / revise",
        ha="center",
        fontsize=5.1,
        color=B4_COLOR,
        transform=ax.transAxes,
    )
    ax.text(
        0.78,
        0.335,
        "report or safe abstain",
        fontsize=4.8,
        color=MID,
        transform=ax.transAxes,
    )


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    draw(ax)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.03)
    save_publication_figure(fig, Path(__file__).with_name("panel_a_system_contrast"))
    plt.close(fig)


if __name__ == "__main__":
    main()
