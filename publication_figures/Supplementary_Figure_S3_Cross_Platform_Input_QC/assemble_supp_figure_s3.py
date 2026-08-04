from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from panel_a_cohort_flow.plot_panel_a import draw as draw_a
from panel_b_probe_coverage.plot_panel_b import draw as draw_b
from panel_c_raw_expression_space.plot_panel_c import draw as draw_c
from panel_d_standardized_expression_space.plot_panel_d import draw as draw_d
from panel_e_score_distribution.plot_panel_e import draw as draw_e
from supp_s3_style import set_style


ROOT = Path(__file__).resolve().parent


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    set_style()
    fig = plt.figure(figsize=(7.25, 5.15), facecolor="white")
    grid = fig.add_gridspec(2, 12, height_ratios=[0.80, 1.0], hspace=0.48, wspace=0.92)
    draw_a(fig.add_subplot(grid[0, :8]), "a")
    draw_b(fig.add_subplot(grid[0, 8:]), "b")
    draw_c(fig.add_subplot(grid[1, :4]), "c")
    draw_d(fig.add_subplot(grid[1, 4:8]), "d")
    draw_e(fig.add_subplot(grid[1, 8:]), "e")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.945, bottom=0.095)
    stem = ROOT / "Supplementary_Figure_S3_Cross_Platform_Input_QC"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
