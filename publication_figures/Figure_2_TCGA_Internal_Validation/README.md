# Figure 2 — TCGA-LIHC internal validation

This directory is self-contained for the internal-validation main figure.

## Build

Run, in order:

1. `python figure2_analysis.py`
2. each `panel_*/plot_panel_*.py` for standalone panels
3. `python assemble_figure2.py`
4. `python verify_figure2.py`

## Panel directories

- `panel_a_model_discrimination`: 25-fold Harrell and Uno C distributions.
- `panel_b_paired_differences`: canonical v6 paired bootstrap comparisons.
- `panel_c_oof_survival`: patient-level OOF KM curves, at-risk table and statistics.
- `panel_d_prediction_error`: paired Brier scores and IBS summary.
- `panel_e_sensitivity`: SA1–SA3 metric values and ranks.

Each panel directory contains clean source data, plotting code and standalone
SVG/PDF/PNG/TIFF exports. The assembled figure is exported in the root directory.

## Interpretation

M4 is the provisional primary model because it has the strongest overall internal
performance and sensitivity profile. The figure deliberately preserves the key
statistical limitation: M4 is not significantly better than M1 after the
pre-specified correction.

The existing `Figure_2_Phase4_Benchmark` directory is intentionally untouched.
Renumbering it should occur only after this new Figure 2 is accepted.
