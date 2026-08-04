# Table 3 QA report

- Overall status: **TABLE3_QA_PASSED**
- Included cohorts: GSE14520 GPL3921 and GSE116174 GPL570.
- Total analysis population: 285 patients and 112 deaths.
- Bootstrap intervals: 1,000/1,000 valid draws for each metric.
- Frozen TCGA threshold applied unchanged in both cohorts.
- GPL571 recorded as not analysed (N=21).
- Claim boundary: gene-only cross-platform transport, not M4 validation.

## Automated checks

- PASS: workbook_exists
- PASS: preview_exists
- PASS: cohorts_exact
- PASS: total_patients_285
- PASS: total_events_112
- PASS: all_bootstrap_iterations_1000
- PASS: high_low_counts_match_cohort_size
- PASS: single_frozen_cutoff
- PASS: no_external_outcome_grouping
- PASS: no_external_recalibration
- PASS: gpl571_not_in_performance
- PASS: gpl571_exclusion_recorded
- PASS: two_nonzero_genes
- PASS: nonzero_genes_exact
- PASS: source_gate_locked
- PASS: formula_errors_absent
