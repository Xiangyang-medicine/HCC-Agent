from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from figure2_v2_style import set_style
from panel_a_system_contrast.plot_panel_a import draw as draw_a
from panel_b_primary_endpoint.plot_panel_b import draw as draw_b
from panel_c_ablation_effects.plot_panel_c import draw as draw_c
from panel_d_traceability_reliability.plot_panel_d import draw as draw_d
from panel_e_fault_handling.plot_panel_e import draw as draw_e


ROOT = Path(__file__).resolve().parent
FIGURE_STEM = ROOT / "Figure_2_Phase4_Benchmark_v2"


def main() -> None:
    # Keep the publication-critical settings explicit in the canonical assembler
    # so the source can be audited without resolving helper-module imports.
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
    fig = plt.figure(figsize=(7.25, 6.15), facecolor="white")
    grid = fig.add_gridspec(
        3,
        12,
        height_ratios=[1.00, 1.00, 1.22],
        hspace=0.48,
        wspace=0.60,
    )
    ax_a = fig.add_subplot(grid[0, :6])
    ax_b = fig.add_subplot(grid[0, 6:])
    ax_c = fig.add_subplot(grid[1, :5])
    ax_d = fig.add_subplot(grid[1, 5:])
    ax_e = fig.add_subplot(grid[2, :])

    draw_a(ax_a, "a")
    draw_b(ax_b, "b")
    draw_c(ax_c, "c")
    draw_d(ax_d, "d")
    draw_e(ax_e, "e")

    fig.subplots_adjust(left=0.155, right=0.985, top=0.973, bottom=0.075)
    export_padding = 0.02
    fig.savefig(
        FIGURE_STEM.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=export_padding,
    )
    fig.savefig(
        FIGURE_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=export_padding,
    )
    fig.savefig(
        FIGURE_STEM.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=export_padding,
    )
    fig.savefig(
        FIGURE_STEM.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=export_padding,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
