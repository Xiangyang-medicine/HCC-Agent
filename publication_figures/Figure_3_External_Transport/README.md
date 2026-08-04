# Figure 3 — external cross-platform transport

This is the manuscript Figure 3 package for the canonical Phase 3B secondary
microarray transport analysis.

## Outputs

- `Figure_3_External_Transport.svg` — editable vector master.
- `Figure_3_External_Transport.pdf` — submission vector.
- `Figure_3_External_Transport.tiff` — 600-dpi LZW raster.
- `Figure_3_External_Transport.png` — 300-dpi preview.

## Panel folders

- `panel_a_transport_workflow/` — external Kaplan–Meier curves using the
  frozen TCGA-derived median risk threshold. The previous cohort-flow data are
  retained as `supplementary_cohort_flow.csv` and are not displayed in the
  main figure.
- `panel_b_cohort_flow/` — continuous transported-score association with OS.
- `panel_c_frozen_coefficients/`
- `panel_d_external_discrimination/`

Each panel folder contains `source_data.csv`, its plotting script and standalone
SVG/PDF/TIFF/PNG exports.

## Reproduction

1. Run `python figure3_analysis.py`.
2. Run each panel plotting script.
3. Run `python assemble_figure3.py`.
4. Run `python verify_figure3.py`.

The expected terminal gate is `FIGURE3_QA_GATE.json` with
`status=FIGURE3_QA_PASSED`.
