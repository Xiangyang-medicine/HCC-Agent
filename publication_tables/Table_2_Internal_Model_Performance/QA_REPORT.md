# Table 2 QA report

- Overall status: **TABLE2_QA_PASSED**
- Workbook: `Table_2_Internal_Model_Performance.xlsx`
- Models: M1, M2, M3, M4, M5
- Formal paired comparisons: 8
- Canonical comparison version: v6
- M4 descriptive status: best Harrell C, Uno C, 36-month AUC, and IBS.
- Multiplicity-aware conclusion: M4 was not significantly better than M1.
- M5 was significantly worse than M1 for Uno C after Bonferroni correction.

## Automated checks

- PASS: workbook_exists
- PASS: preview_exists
- PASS: model_ids_exact
- PASS: all_models_have_25_folds
- PASS: formal_comparisons_exactly_8
- PASS: all_bootstraps_complete
- PASS: m4_best_harrell_descriptively
- PASS: m4_best_uno_descriptively
- PASS: m4_lowest_ibs_descriptively
- PASS: m4_harrell_vs_m1_not_significant_adjusted
- PASS: m4_uno_vs_m1_not_significant_adjusted
- PASS: m5_uno_vs_m1_significantly_worse
- PASS: m4_harrell_greater_than_m1_descriptively
- PASS: source_gate_locked
- PASS: formula_errors_absent
