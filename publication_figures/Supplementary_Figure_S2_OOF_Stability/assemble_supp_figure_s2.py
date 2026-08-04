from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from panel_a_repeat_correlation.plot_panel_a import draw as draw_a
from panel_b_patient_rank_dispersion.plot_panel_b import draw as draw_b
from panel_c_m4_quintile_transition.plot_panel_c import draw as draw_c
from panel_d_m5_quintile_transition.plot_panel_d import draw as draw_d
from panel_e_consensus_and_qc.plot_panel_e import draw as draw_e
from supp_s2_style import set_style


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
    fig = plt.figure(figsize=(7.25, 6.10), facecolor="white")
    grid = fig.add_gridspec(3, 12, height_ratios=[1.0, 1.05, 0.83], hspace=0.68, wspace=0.78)
    draw_a(fig.add_subplot(grid[0, :6]), "a")
    draw_b(fig.add_subplot(grid[0, 6:]), "b")
    draw_c(fig.add_subplot(grid[1, :6]), "c")
    draw_d(fig.add_subplot(grid[1, 6:]), "d")
    draw_e(fig.add_subplot(grid[2, 1:10]), "e")
    fig.subplots_adjust(left=0.09, right=0.92, top=0.955, bottom=0.085)
    stem = ROOT / "Supplementary_Figure_S2_OOF_Stability"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
