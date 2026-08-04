from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figure2_style import save_publication_figure, set_style
from panel_a_primary_success.plot_panel_a import draw as draw_a
from panel_b_functional_decomposition.plot_panel_b import draw as draw_b
from panel_c_grounding_safety.plot_panel_c import draw as draw_c
from panel_d_ablation.plot_panel_d import draw as draw_d
from panel_e_reliability_efficiency.plot_panel_e import draw as draw_e
from panel_f_fault_matrix.plot_panel_f import draw as draw_f


ROOT = Path(__file__).resolve().parent


def main() -> None:
    set_style()
    fig = plt.figure(figsize=(7.25, 9.20), facecolor="white")
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.18, 1.24, 1.25, 1.72],
        hspace=0.55,
        wspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_f = fig.add_subplot(grid[2, 1])
    ax_e = fig.add_subplot(grid[3, :])

    draw_a(ax_a, "a")
    draw_b(ax_b, "b")
    draw_c(ax_c, "c")
    draw_d(ax_d, "d")
    draw_e(ax_f, "e")
    draw_f(ax_e, "f")

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.055, top=0.975)
    save_publication_figure(fig, ROOT / "Figure_2_Phase4_Benchmark")
    plt.close(fig)


if __name__ == "__main__":
    main()
