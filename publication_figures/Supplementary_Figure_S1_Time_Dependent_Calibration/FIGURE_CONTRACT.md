# Supplementary Figure S1 contract

## Core conclusion

Repeated OOF predictions show time-dependent discrimination across 12, 36 and 60 months, while Kaplan–Meier calibration curves characterize how the selected M4 model behaves relative to the clinical M1 reference without reducing diagnostics to numeric grids.

## Evidence map

- **a:** Time-dependent AUC trajectories for M1–M5 across all 25 outer test folds.
- **b:** 12-month OOF calibration curves for M1 and M4.
- **c:** 36-month OOF calibration curves for M1 and M4.
- **d:** 60-month OOF calibration curves for M1 and M4.

## Calibration construction

- Each repeat is analysed separately; the five predictions per patient are never treated as five independent patients.
- Within each model, repeat and horizon, patients are grouped into six equal-frequency bins of predicted event risk.
- Observed event probability is `1 − Kaplan–Meier survival` at the fixed horizon.
- Light lines show the five repeat-specific curves. Dark lines and points show across-repeat means by risk bin.
- The identity line indicates perfect calibration.
- Curves are descriptive; no calibration hypothesis test is performed.

## Source boundary

- `F:\ACM\experiments\phase3a\formal\metrics_summary.json`
- `F:\ACM\experiments\phase3a\formal\oof_predictions.csv`

## Output

7.25 × 5.15 inches, editable SVG/PDF and 600-dpi TIFF.
