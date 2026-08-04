# Figure 4 quality-assurance report

Status: `FIGURE4_QA_PASSED`

## Data and methods

- Complete frozen formal benchmark: 4,860/4,860 records.
- Primary comparison: 300 paired runs per system.
- Primary absolute difference: +13.0 percentage points.
- Post-hoc strict structural audit: 0/600 B2–B4 clean-run disagreements.
- Prespecified confirmatory endpoint modified: no.
- `schema_valid` is absent from the main panels.
- The B4-only internal verifier is not encoded as a zero-valued comparator.
- Numeric source tables contain no infinite values.

## Figure outputs

- SVG contains editable text and all five panel labels.
- PDF is vector.
- PNG: 2,144 × 1,761 pixels at 300 dpi.
- TIFF: 4,291 × 3,526 pixels at 600 dpi, LZW compressed.
- Nominal canvas: 7.25 × 6.15 inches.

## Automated checks

- Figure-specific verification: 25/25 checks passed.
- Static source preflight: 14 PASS, 0 WARN, 0 FAIL.
- Python source parses successfully.
- Publication-safe sans-serif font stack is configured.
- SVG and PDF editable-text settings are configured.
- SVG, PDF, PNG and TIFF export paths are present.
- No rainbow colour map, simulated-data generator, row sampling, unreported
  missing-data exclusion or cross-backend plotting reference was detected.

## Visual inspection

The final 300 dpi preview was inspected at full resolution. Panel headings,
axis labels, confidence intervals, direct labels, the paired-run inset and the
fault-handling legend are legible without overlap or clipping. The colour
encoding is reinforced by labels, marker shapes and panel structure.

## Reproducibility

Each panel folder contains its source data and plotting code. The root folder
contains the offline audit, shared style module, canonical assembler, figure
contract, legend, static preflight report and machine-readable QA gate.
