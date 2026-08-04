# Supplementary Figure S2 contract

## Core conclusion

Repeated out-of-fold predictions show that M4 preserves patient risk ranking across resamples, whereas M5 is unstable. This descriptive stability analysis supports, but does not replace, the prespecified performance-based candidate selection.

## Evidence map

- **a** Pairwise Spearman correlations of within-repeat OOF risk percentiles across all ten repeat pairs.
- **b** Patient-level standard deviation of OOF risk percentiles across five repeats.
- **c** Joint density of patient risk percentiles across all ten repeat pairs for M4.
- **d** Joint density of patient risk percentiles across all ten repeat pairs for M5.
- **e** Modal quintile agreement across five repeats for every model, with structural survival-probability checks.

## Statistical stance

- All analyses are descriptive.
- Risk scores are converted to within-model, within-repeat percentiles before cross-repeat comparison because raw score scales can differ after fold-specific fitting.
- No fold, patient, or model is excluded.
- No inferential p-values are reported.
- Density panels include 3,630 paired observations per model (363 patients × 10 repeat pairs).

## Source boundary

Canonical input: `F:\ACM\experiments\phase3a\formal\oof_predictions.csv` (9,075 rows; 363 patients × 5 repeats × 5 models).

## Output

Final size: 7.25 × 6.10 inches (approximately 184 × 155 mm), with editable SVG/PDF and 600-dpi TIFF.
