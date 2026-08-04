from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from panel_a_benchmark_composition.plot_panel_a import draw as draw_a
from panel_b_paired_case_success.plot_panel_b import draw as draw_b
from panel_c_repeat_reliability.plot_panel_c import draw as draw_c
from panel_d_verification_repair_flow.plot_panel_d import draw as draw_d
from panel_e_fault_outcomes.plot_panel_e import draw as draw_e
from panel_f_support_precision_completeness.plot_panel_f import draw as draw_f
from supp_s4_style import set_style


ROOT = Path(__file__).resolve().parent


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    set_style()
    fig = plt.figure(figsize=(7.25, 7.30), facecolor="white")
    grid = fig.add_gridspec(3, 12, height_ratios=[0.83, 0.83, 1.15], hspace=0.72, wspace=0.82)
    draw_a(fig.add_subplot(grid[0, :6]), "a")
    draw_b(fig.add_subplot(grid[0, 6:]), "b")
    draw_c(fig.add_subplot(grid[1, :6]), "c")
    draw_d(fig.add_subplot(grid[1, 6:]), "d")
    draw_e(fig.add_subplot(grid[2, :8]), "e")
    draw_f(fig.add_subplot(grid[2, 8:]), "f")
    fig.subplots_adjust(left=0.11, right=0.975, top=0.965, bottom=0.105)
    stem = ROOT / "Supplementary_Figure_S4_Agent_Reliability"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
