from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_panel import draw_calibration  # noqa: E402
from supp_s1_style import set_style  # noqa: E402


def draw(ax: plt.Axes, letter: str = "b") -> None:
    draw_calibration(ax, Path(__file__).with_name("source_data.csv"), letter, 12)


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 6.6, "svg.fonttype": "none", "pdf.fonttype": 42})
    set_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    draw(ax)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.81, bottom=0.19)
    stem = Path(__file__).with_name("panel_b_calibration_12m")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
