# Table 4 QA report

- Overall status: **TABLE4_QA_PASSED**
- Formal records: 4,860/4,860 unique; 0 API errors.
- Primary B2 result: 245/300 (81.7%).
- Primary B4 result: 284/300 (94.7%).
- Primary difference: +13.0 percentage points.
- Strict post-hoc report-contract audit: 0/600 clean-run disagreements.
- Main table excludes the misleading `schema_valid` label.
- B2 internal verifier is N/A, not zero.

## Automated checks

- PASS: workbook_exists
- PASS: preview_exists
- PASS: formal_record_count_4860
- PASS: formal_unique_count_4860
- PASS: no_api_errors
- PASS: systems_exact_b2_b4
- PASS: primary_300_runs_each
- PASS: primary_success_counts
- PASS: primary_rates
- PASS: primary_difference_13pp
- PASS: primary_difference_ci
- PASS: paired_outcomes_sum_300
- PASS: traceability_rows_6
- PASS: planning_rows_2
- PASS: four_ablations
- PASS: eight_fault_types
- PASS: unsupported_scoring_mismatch_preserved
- PASS: confirmatory_endpoint_unchanged
- PASS: strict_audit_zero_disagreement
- PASS: schema_metric_absent
- PASS: b2_verifier_not_zero_coded
- PASS: source_gate_locked
- PASS: formula_errors_absent
