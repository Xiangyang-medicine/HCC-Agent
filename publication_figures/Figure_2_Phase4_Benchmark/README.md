# Figure 2 package

The manuscript-canonical version is now `revision_v2/`. It preserves the frozen
formal benchmark, corrects the metric terminology, adds an offline structural
scoring audit, and replaces the dashboard-style layout in this parent
directory.

The files and panel folders listed below are retained only as the superseded
v1 package for provenance. Do not use them in the manuscript.

## Superseded v1 assembled figure

- `Figure_2_Phase4_Benchmark.svg` — editable vector master.
- `Figure_2_Phase4_Benchmark.pdf` — embedded-font vector copy.
- `Figure_2_Phase4_Benchmark.tiff` — 600-dpi LZW submission raster.
- `Figure_2_Phase4_Benchmark.png` — 300-dpi review preview.

## Panel folders

- `panel_a_primary_success/`
- `panel_b_functional_decomposition/`
- `panel_c_grounding_safety/`
- `panel_d_ablation/`
- `panel_e_reliability_efficiency/`
- `panel_f_fault_matrix/`

Each panel folder contains its panel-specific source-data CSV, plotting script,
and SVG/PDF/TIFF/PNG exports. Panel a and d also include their statistical
comparison CSV files.

## Reproduction

1. Run `figure2_analysis.py`.
2. Run each `plot_panel_*.py` script.
3. Run `assemble_figure2.py`.
4. Run `verify_figure2.py`.

The expected terminal gate is `FIGURE2_QA_GATE.json` with
`status=FIGURE2_VERIFIED` and all checks passed.

## Interpretation of the retained v1 package

This figure supports technical claims about verified task completion,
orchestration, evidence traceability, closed-loop verification, fault handling,
repeat reliability, and computational cost within the frozen benchmark. It does
not support clinical deployment or patient-benefit claims.
