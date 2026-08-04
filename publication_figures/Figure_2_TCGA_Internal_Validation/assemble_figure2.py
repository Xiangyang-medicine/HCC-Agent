from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from figure2_style import set_style
from panel_a_model_discrimination.plot_panel_a import draw as draw_a
from panel_b_paired_differences.plot_panel_b import draw as draw_b
from panel_c_oof_survival.plot_panel_c import draw as draw_c
from panel_d_prediction_error.plot_panel_d import draw as draw_d
from panel_e_sensitivity.plot_panel_e import draw as draw_e


ROOT = Path(__file__).resolve().parent
FIGURE_STEM = ROOT / "Figure_2_TCGA_Internal_Validation"


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
    fig = plt.figure(figsize=(7.25, 8.15), facecolor="white")
    grid = fig.add_gridspec(
        3,
        14,
        height_ratios=[1.00, 1.06, 0.74],
        hspace=0.98,
        wspace=1.28,
    )
    ax_a = fig.add_subplot(grid[0, :8])
    ax_b = fig.add_subplot(grid[0, 8:])
    ax_c = fig.add_subplot(grid[1, :8])
    ax_d = fig.add_subplot(grid[1, 8:])
    ax_e = fig.add_subplot(grid[2, :])

    draw_a(ax_a, "a")
    draw_b(ax_b, "b")
    draw_c(ax_c, "c")
    draw_d(ax_d, "d")
    draw_e(ax_e, "e")

    fig.subplots_adjust(left=0.125, right=0.985, top=0.965, bottom=0.042)
    fig.savefig(FIGURE_STEM.with_suffix(".svg"))
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(
        FIGURE_STEM.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
