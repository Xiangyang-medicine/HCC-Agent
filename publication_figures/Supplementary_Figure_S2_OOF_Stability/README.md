# Supplementary Figure S2

This folder contains the complete reproducible artifact for the repeated OOF stability analysis.

## Build order

1. Run `prepare_source_data.py`.
2. Run each panel script in its panel folder.
3. Run `assemble_supp_figure_s2.py`.
4. Run `verify_supp_figure_s2.py`.

All source data derive from the canonical 9,075-row Phase 3A formal OOF prediction file. No simulated data, post-hoc sign flipping, outcome-based filtering, or patient exclusion is used.

The standalone panel widths are intermediate assembly sizes; the combined figure is exported at the journal full-width target.
