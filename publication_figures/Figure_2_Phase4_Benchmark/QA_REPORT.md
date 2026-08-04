# Figure 2 QA report

## Status

`FIGURE2_VERIFIED`: 120/120 machine checks passed. Static plotting preflight:
14/14 passed with no warnings or failures. The assembled PDF is 7.150 × 8.952
inches after tight bounding-box export.

## Data integrity

- Formal records expected: 4,860.
- Clean records: 1,500.
- Ablation records: 1,200.
- Fault records: 2,160.
- Analysis unit for inference: patient `case_id`.
- Formal repeats: three per clean system/case.
- No formal case or failed run was excluded.
- No simulated tutorial data or files from the outside-data folder were used.

## Statistical integrity

- Primary comparison: B4 versus B2 only.
- Confidence intervals: patient-clustered percentile bootstrap, 2,000 resamples.
- Primary test: two-sided paired sign-permutation, 100,000 draws.
- Ablations: paired sign-permutation with Holm correction.
- Secondary heatmaps and grounding rates are descriptive unless explicitly
  identified otherwise.
- Repeated runs are retained within the sampled patient cluster.

## Visual/export integrity

- Backend: Python only.
- Final source size: 7.25 × 9.20 inches.
- Exports: editable SVG, embedded-font PDF, 300-dpi PNG, and 600-dpi LZW TIFF.
- Font family: Arial/Helvetica/sans-serif.
- Minimum configured text size: 6.4 pt.
- No rainbow colour map; text values are printed in every heatmap cell.
- All six panels have panel-specific source data and plotting code.

## Interpretation boundary

The figure evaluates agent orchestration, evidence traceability, verification,
revision, fault handling, reliability, and efficiency under the frozen benchmark.
It does not establish clinical utility, diagnosis, treatment benefit, physician
acceptance, deployment readiness, or improved patient outcomes. B0 has zero
verified task success because the composite endpoint requires agent behaviours
that a non-agent prognostic engine cannot perform.
