# Figure 2 contract — TCGA-LIHC internal validation

## Core claim

Across leakage-controlled 5×5 repeated nested cross-validation in 363 TCGA-LIHC
patients, the combined random-survival-forest model (M4) has the strongest overall
internal performance profile and stable sensitivity results, but its advantage over
the clinical Cox baseline is **not statistically significant after pre-specified
multiple-comparison correction**.

## Archetype and hierarchy

- Archetype: results-first multi-panel validation figure.
- Hero evidence: panel a, outer-fold discrimination distributions.
- Confirmatory evidence: panel b, canonical patient-level paired bootstrap.
- Clinical interpretability: panel c, strictly OOF survival association.
- Complementary performance dimension: panel d, prediction error over time.
- Robustness: panel e, cohort-definition sensitivity analyses.

## Panel contracts

### a — Outer-fold discrimination

- Data: canonical `metrics_summary.json`.
- Split: 5 repeats × 5 outer test folds.
- Metrics: Harrell C and Uno C; Uno C uses fold-specific IPCW.
- Center/spread: all 25 fold estimates shown; horizontal line inside each box is the
  median and the thicker colored line is the arithmetic mean.
- No inferential P values are attached to fold distributions.

### b — Pre-specified paired comparisons

- Data: canonical `model_comparisons_v6.csv` only.
- Sampling unit: patient.
- Procedure: same bootstrap patient sample for both models and all repeats; metric
  differences are computed within repeat and averaged.
- Iterations: 1,000 valid bootstrap samples.
- Interval: percentile 95% confidence interval.
- Correction: Bonferroni correction across four formal comparisons within each
  metric family.
- Direction: positive values favor model A.

### c — Patient-level OOF survival association

- Data: `oof_predictions.csv`, M4 only.
- Each patient has five OOF predictions, one per repeat.
- Because fold-specific RSF risk scales are not directly exchangeable, risk is
  converted to an outcome-blind percentile within each outer test fold and averaged
  across repeats.
- Groups: median split of the patient-level mean OOF percentile.
- Curves: Kaplan–Meier with 95% confidence bands.
- Test: two-sided log-rank test.
- Effect: univariable Cox HR for high versus low OOF risk, with 95% CI.
- Boundary: this is internal OOF association, not an externally validated clinical
  cutoff.

### d — Prediction error

- Data: canonical per-fold Brier scores from `metrics_summary.json`.
- Models: M1 clinical Cox and M4 combined RSF.
- Horizons: 12, 36 and 60 months.
- Points: 25 outer-fold estimates.
- Lines: pair the same repeat/fold.
- Large point/error bar: mean and normal-approximation 95% CI across folds.
- IBS: mean integrated Brier score across 25 folds.

### e — Sensitivity analyses

- Data: canonical `SENSITIVITY_SUMMARY_V2.csv`.
- SA1: primary cohort, N=363.
- SA2: patients aged <18 years excluded, N=361.
- SA3: complete stage and grade, N=338.
- Cell text: metric value.
- Cell shading: within-analysis rank; for IBS, lower is better.

## Visual system

- Final width: 7.25 inches (approximately 184 mm, double column).
- Font: Arial/Helvetica-compatible sans serif; editable SVG/PDF text.
- Panel labels: lowercase bold.
- Color: model identity uses a color-blind-aware navy/blue/teal/rust palette;
  numeric labels and shape/position prevent color-only interpretation.
- Export: SVG, PDF, 300-dpi PNG and 600-dpi LZW TIFF.

## Prohibited claims

- Do not state that M4 is statistically superior to M1 after correction.
- Do not call the median OOF split a validated clinical threshold.
- Do not mix or pool repeated OOF predictions as independent patients.
- Do not use `model_comparisons_v5` or earlier comparison files.
