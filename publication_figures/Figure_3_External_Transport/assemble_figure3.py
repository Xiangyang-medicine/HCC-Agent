from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from figure3_style import set_style
from panel_a_transport_workflow.plot_panel_a import draw as draw_a
from panel_b_cohort_flow.plot_panel_b import draw as draw_b
from panel_c_frozen_coefficients.plot_panel_c import draw as draw_c
from panel_d_external_discrimination.plot_panel_d import draw as draw_d


ROOT = Path(__file__).resolve().parent
FIGURE_STEM = ROOT / "Figure_3_External_Transport"


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 6.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    set_style()
    fig = plt.figure(figsize=(7.25, 5.85), facecolor="white")
    grid = fig.add_gridspec(
        2,
        12,
        height_ratios=[0.92, 1.30],
        hspace=0.42,
        wspace=0.95,
    )
    ax_a = fig.add_subplot(grid[0, :7])
    ax_b = fig.add_subplot(grid[0, 7:])
    ax_c = fig.add_subplot(grid[1, :5])
    ax_d = fig.add_subplot(grid[1, 5:])

    draw_a(ax_a, "a")
    draw_b(ax_b, "b")
    draw_c(ax_c, "c")
    draw_d(ax_d, "d")

    fig.subplots_adjust(left=0.12, right=0.985, top=0.905, bottom=0.080)
    fig.savefig(FIGURE_STEM.with_suffix(".svg"))
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(
        FIGURE_STEM.with_suffix(".png"),
        dpi=300,
    )
    fig.savefig(
        FIGURE_STEM.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
