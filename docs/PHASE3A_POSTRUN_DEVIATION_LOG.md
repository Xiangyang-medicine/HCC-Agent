# Phase 3A Post-Run Deviation Log

**Created:** 2026-07-21
**Last Updated:** 2026-07-24 (v1.8)
**Status:** PHASE3A_METHOD_CLOSURE_COMPLETED
**Maintainers:** Research team audit trail (Claude Code implementation; Codex v6 review and correction)

---

## 1. Status Correction

### Current State
- **Status Code:** `PHASE3A_METHOD_CLOSURE_COMPLETED`
- **Authoritative Outputs:** `model_comparisons_v6.csv`, `model_comparisons_v6.json`, and `AUDIT_REPORT_V5.json`
- **Meaning:** v6 comparisons use patient-clustered paired bootstrap. Uno C is recomputed in all 25 locked outer folds with IPCW and tau derived only from the corresponding outer-training cohort.
- **M4 Status:** PROVISIONAL_PRIMARY_CANDIDATE (NOT significant vs M1 on either metric)

### Methodological Issues Identified
1. **Model Comparisons (v1)**: Fold-level bootstrap (25 folds) instead of patient-level (363 patients) - **RESOLVED v2**
2. **PH Diagnostics (v1)**: Fisher's method ≠ standard global Schoenfeld test; misleading claim - **RESOLVED v2**
3. **Sensitivity Analysis**: No proper entry point for SA2/SA3 execution - **RESOLVED**
4. **Uno C v4/v5**: IPCW was not estimated from each outer-training fold; v5 also queried train-indexed weights with test IDs and allowed a zero bootstrap p-value - **RESOLVED v6**

### Superseded Files (NOT FOR PUBLICATION)
The following files have been superseded due to methodological issues and MUST NOT be used for publication:

| Original File | Superseded File | Reason |
|--------------|-----------------|--------|
| `model_comparisons.csv` | `model_comparisons_SUPERSEDED.csv` | Fold-level bootstrap, not patient-level |
| `model_comparisons.json` | `model_comparisons_SUPERSEDED.json` | Fold-level bootstrap, not patient-level |
| `ph_diagnostics.csv` | `ph_diagnostics_SUPERSEDED.csv` | Fisher's method ≠ global Schoenfeld test |
| `ph_diagnostics_summary.json` | `ph_diagnostics_summary_SUPERSEDED.json` | Misleading "25/25 passed" claim |
| `model_comparisons_v4.*` | superseded by `model_comparisons_v6.*` | Used test data as an IPCW proxy and permitted partial-fold bootstrap iterations |
| `model_comparisons_v5.*` | superseded by `model_comparisons_v6.*` | Incorrect IPCW lookup, fold-1 hardcoding, and bootstrap p-value without finite-sample correction |
| `AUDIT_REPORT_V3/V4.json` | superseded by `AUDIT_REPORT_V5.json` | Audited invalid v4/v5 comparison implementations |

### Completed Tasks (2026-07-21)
- PH diagnostics (v1): **25/25 folds PASSED** (all global p > 0.05)
- Per-covariate: `stage_nan` shows 7/25 violations (mean p=0.27) but global test still passes

### Version 1.3 Completions (2026-07-23)
- **Model Comparisons v2**: Patient-level bootstrap (1000 iterations, 363 patients) ✓ COMPLETED
- **PH Diagnostics v2**: Proper per-covariate reporting with Bonferroni correction ✓ COMPLETED
- **Critical Finding**: M4 vs M1 NOT statistically significant (p_adj = 0.080 > 0.0125 Bonferroni threshold)

### Protected Files (PROHIBITED from overwrite/delete)
- `experiments/phase3a/formal/oof_predictions.csv` - 9075 predictions
- `experiments/phase3a/formal/metrics_summary.json` - Complete metrics
- `experiments/phase3a/formal/AUDIT_REPORT.json` - Original audit (NOT to be modified)
- `experiments/phase3a/formal/logs/` - Training logs

**Superseded Files (SUPERSEDED_NOT_FOR_PUBLICATION):**
- `ph_diagnostics.csv` → `ph_diagnostics_SUPERSEDED.csv`
- `ph_diagnostics_summary.json` → `ph_diagnostics_summary_SUPERSEDED.json`
- `model_comparisons.csv` → `model_comparisons_SUPERSEDED.csv`
- `model_comparisons.json` → `model_comparisons_SUPERSEDED.json`
- `model_comparisons_v2.csv` → `model_comparisons_v2_SUPERSEDED.csv`
- `model_comparisons_v2.json` → `model_comparisons_v2_SUPERSEDED.json`

### Original File Hashes (for integrity tracking)
| File | SHA-256 |
|------|---------|
| oof_predictions.csv | 7b21074e208a563bc99b6a0a8c458f076a5ab333612e65ce2659cd9d6571228f |
| metrics_summary.json | 1874ac2c7b94f5724b95de7316b7eebe1cc2f32c0a80fa306fb82510d6da8e48 |

---

## 2. Formal Results Status

### PENDING CORRECTIONS ✗
The following files require methodological correction:
- `model_comparisons_v2.*` - Patient-level bootstrap (NOT fold-level)
- `ph_diagnostics_v2.*` - Proper global test reporting (NOT Fisher's method)

### Completed Tasks (underlying data) ✓
- Formal prediction structure: **16/16 checks PASSED**
- SHA-256 verification of locked files: **5/5 PASSED**
- pytest: **66/66 tests PASSED**
- Gate files verification: **PASS**
- Training execution: **COMPLETED** (5 repeats × 5 folds = 25 folds)
- 9075 OOF predictions generated
- All 5 models completed (M1-M5)
- OOF metrics available for proper statistical comparison

### NOT YET COMPLETED ✗
1. **Sensitivity Analyses** - SA2/SA3 datasets prepared but training not executed
2. **Phase 3B Protocol Amendment** - Required since M4 selected (RSF) for external validation

### Status Update (2026-07-21 afternoon)
- M5 Diagnostic Report: **COMPLETED** - See `docs/PHASE3A_M5_DIAGNOSTIC_REPORT.md`
- Candidate Model Selection: **COMPLETED** - See `docs/PHASE3A_CANDIDATE_MODEL_SELECTION.md`
- M4 (Combined RSF) selected as primary candidate

### Critical Note on AUDIT_REPORT.json
The field `all_validations_passed: true` in AUDIT_REPORT.json **DOES NOT** represent completion of the full SAP delivery. It only represents the formal prediction pipeline integrity checks passing. The following must still be completed:
- PH diagnostics
- Prespecified model comparisons (4 formal + 1 exploratory)
- Sensitivity analyses (SA2, SA3)
- M5 diagnostic report
- Candidate model selection
- Phase 3B protocol review

---

## 3. Known Issues

### Issue 1: PH Diagnostics - RESOLVED
**Error:** `shapes (290/291, 19) and (15-18,) not aligned`

**Root Cause:** M1ClinicalCox removes low-variance columns during fit, but PH test used original column set.

**Resolution:** Created `run_ph_diagnostics.py` with fixed column alignment using model's actual feature names. All 25 folds PASSED.

**Minor Finding:** `stage_nan` covariate shows 7/25 violations (mean p=0.27), suggesting missing stage data may violate PH. This is a data quality concern but does not invalidate the overall model.

---

## 4. Protocol Notes

### SAP Version Conflict
The original SAP specifies "paired t-test" for model comparisons. Subsequent revisions adopted "patient-clustered Bootstrap." This deviation log records this conflict. Both methods will be reported:
- Patient-clustered Bootstrap as primary (robust analysis)
- Paired t-test as supplementary (as specified in original SAP)

---

## 5. Phase 3B Blocking Status

**STATUS:** SA2/SA3 COMPLETED as of 2026-07-24. Phase 3B protocol amendment pending.

| Task | Status | Notes |
|------|--------|-------|
| Corrected PH diagnostics (v2) | ✓ DONE | Per-covariate with Bonferroni correction |
| Corrected model comparisons (v2) | ✓ DONE | Patient-level bootstrap (17 tests pass) |
| Sensitivity analysis framework | ✓ DONE | SA2/SA3 smoke tests passed |
| SA2 full training (9025 predictions) | ✓ DONE | 361 patients, Harrell C = 0.636 |
| SA3 full training (8450 predictions) | ✓ DONE | 338 patients, Harrell C = 0.628 |
| SENSITIVITY_SUMMARY generated | ✓ DONE | M4 consistently #1 across all analyses |
| Phase 3B protocol amendment | ✗ PENDING | Required for M4 RSF external validation |
| M4 as PROVISIONAL primary candidate | ✓ DONE | NOT significant vs M1 (p_adj=0.080) |

**Remaining blockers for Phase 3B:** Phase 3B protocol amendment

---

## 7. Version 1.3: Corrected Model Comparisons v2

### Key Results (Patient-Level Bootstrap, n=363 patients)

| Comparison | Type | Mean Diff | 95% CI | Raw p | Adj p | Significant |
|------------|------|-----------|--------|-------|-------|-------------|
| M3 vs M1 | Formal | +0.051 | [-0.010, 0.114] | 0.094 | 0.376 | **No** |
| M4 vs M1 | Formal | +0.077 | [0.012, 0.136] | 0.020 | 0.080 | **No** |
| M5 vs M1 | Formal | -0.079 | [-0.147, -0.012] | 0.020 | 0.080 | **No** |
| M3 vs M2 | Formal | +0.001 | [-0.016, 0.018] | 0.861 | 1.000 | **No** |
| M4 vs M2 | Exploratory | +0.027 | [-0.012, 0.063] | 0.186 | 0.186 | No |

**Bonferroni-corrected threshold:** p < 0.0125 for 4 formal comparisons

### Critical Finding
M4 vs M1 is **NOT statistically significant** at Bonferroni threshold:
- p_adj = 0.080 > 0.0125
- This differs from superseded analysis which incorrectly reported p = 0.0045

### Methodology
- Patient-level paired bootstrap (n=363 patients sampled with replacement)
- 1000 bootstrap iterations
- Finite-sample corrected p-value: `p = min(1, 2 * (min(n_le_0, n_ge_0) + 1) / (n_valid + 1))`
- All raw p-values > 0 (no numerical issues)

### Output Files
- `model_comparisons_v2.csv` - 5 comparisons
- `model_comparisons_v2.json` - Full results with bootstrap distributions

---

## 8. Version 1.3: Corrected PH Diagnostics v2

### Key Results

| Metric | Value |
|--------|-------|
| diagnostics_executed | 25 |
| diagnostics_errors | 0 |
| folds_with_any_covariate_violation | 7 |
| folds_without_detected_violation | 18 |

### Per-Covariate Summary (across 25 folds)

| Covariate | Raw Violations | Bonferroni Violations |
|-----------|----------------|----------------------|
| stage_nan | 7/25 | 0/25 |
| All others | 0/25 | 0/25 |

### Methodology Note
- Per-covariate PH tests with Bonferroni correction
- **Standard global Schoenfeld test not available for penalized Cox models**
- Fisher's method on per-covariate p-values is **NOT equivalent** to global test
- Cannot claim "X/X PH satisfied" based on Fisher's method

### Output Files
- `ph_diagnostics_v2.csv` - 25 fold results
- `ph_diagnostics_v2.json` - Full results with per-covariate p-values

---

## 9. Version 1.4: SA Framework Implementation (2026-07-24)

### SA-Aware Validation

The training module now supports sensitivity analysis with SA-aware validation:

**SA_CONFIG:**
```python
SA_CONFIG = {
    'SA1': {'n_patients': 363, 'predictions_per_model': 1815, 'total': 9075},
    'SA2': {'n_patients': 361, 'predictions_per_model': 1805, 'total': 9025},
    'SA3': {'n_patients': 338, 'predictions_per_model': 1690, 'total': 8450},
}
```

### Smoke Test Results

| Analysis | Patients | Expected | Status |
|----------|---------|----------|--------|
| SA2 | 361 | 1805 | PASSED (PILOT_COMPLETED) |
| SA3 | 338 | 1690 | PASSED (PILOT_COMPLETED) |
| SA1 (Formal) | 363 | 1815 | PASSED (PILOT_COMPLETED) |

### Full Training Status (COMPLETED 2026-07-24)
- SA2: ✓ COMPLETED (9025 predictions, 361 patients, Harrell C = 0.636)
- SA3: ✓ COMPLETED (8450 predictions, 338 patients, Harrell C = 0.628)

---

## 10. Version 1.5: SA2/SA3 Sensitivity Analysis Results

### Analysis Overview

| Analysis | Patients | Events | Exclusion Criteria |
|----------|----------|--------|-------------------|
| SA1 (Formal) | 363 | ~3225 | None |
| SA2 | 361 | ~3225 | 2 pediatric (age < 18) |
| SA3 | 338 | ~2825 | 25 missing stage/grade |

### Model Performance Comparison (Harrell C-index)

| Model | SA1 | SA2 | SA3 | Mean | Rank Stability |
|-------|-----|-----|-----|------|----------------|
| M1 (Clinical Cox) | 0.557 | 0.559 | 0.568 | 0.562 | R4→R4→R4 |
| M2 (Gene Elasticnet) | 0.611 | 0.621 | 0.608 | 0.613 | R2→R2→R2 |
| M3 (Combined Elasticnet) | 0.615 | 0.615 | 0.605 | 0.612 | R3→R3→R3 |
| **M4 (Combined RSF)** | **0.638** | **0.636** | **0.628** | **0.634** | **R1→R1→R1** |
| M5 (DeepSurv) | 0.489 | 0.491 | 0.508 | 0.496 | R5→R5→R5 |

### Key Findings

1. **M4 (Combined RSF) is consistently the best model** across all three sensitivity analyses
2. **Rank stability**: M4 ranked #1 in SA1, SA2, and SA3
3. **Performance range**: M4 Harrell C varies from 0.628 to 0.638 across analyses
4. **Model ordering**: M4 > M2/M3 > M1 > M5 is consistent across all analyses
5. **M5 (DeepSurv) remains unstable**: Near random performance in all analyses

### SENSITIVITY_SUMMARY Files
- `experiments/phase3a/sensitivity/SENSITIVITY_SUMMARY.json` - Full JSON summary
- `experiments/phase3a/sensitivity/SENSITIVITY_SUMMARY.csv` - CSV comparison table

---

## 11. Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2026-07-21 | Initial deviation log created |
| 1.1 | 2026-07-21 | Added M5 diagnostic report and candidate selection completion |
| 1.2 | 2026-07-23 | Status corrected to METHOD_CLOSURE_BLOCKED. Superseded files marked. |
| 1.3 | 2026-07-23 | v2 methodology completed. M4 vs M1 NOT significant (p=0.080). SA framework ready. |
| 1.4 | 2026-07-24 | SA-aware validation implemented. SA2/SA3 smoke tests passed. |
| 1.5 | 2026-07-24 | SA2/SA3 full training completed. M4 consistently #1. SENSITIVITY_SUMMARY generated. |
| 1.6 | 2026-07-24 | model_comparisons_v3 generated. Uno C added. pytest 66/66 passed. |
| 1.7 | 2026-07-24 | model_comparisons_v4: True Uno C bootstrap implemented. M5 significant on Uno C only. |

---

## 12. Version 1.6: Model Comparisons v3 (2026-07-24)

### Key Changes from v2
- Added Uno C-index comparisons (supplementary t-test only, IPCW training data not available)
- Per-fold t-test from metrics_summary.json (not OOF mixing)
- Both Harrell C (primary bootstrap) and Uno C (supplementary t-test) reported

### Output Files
- `model_comparisons_v3.csv` - 10 comparisons (5 Harrell C + 5 Uno C)
- `model_comparisons_v3.json` - Full results

### Key Results

**Harrell C-index (Patient-Level Bootstrap):**
| Comparison | Type | M4 Better | p_adj | Significant |
|------------|------|-----------|-------|-------------|
| M4 vs M1 | Formal | M4 (+0.044) | 0.096 | No |
| M3 vs M1 | Formal | M3 (+0.026) | 0.488 | No |
| M5 vs M1 | Formal | M1 (-0.118) | <0.001 | **Yes** |

**Uno C-index (Supplementary T-Test, Per-Fold n=25):**
| Comparison | Type | M4 Better | p_adj | Significant |
|------------|------|-----------|-------|-------------|
| M4 vs M1 | Formal | M4 (+0.012) | 1.000 | No |
| M5 vs M1 | Formal | M1 (-0.129) | <0.001 | **Yes** |

### Critical Finding
M4 vs M1 is **NOT statistically significant** at Bonferroni-corrected threshold (0.0125) for either metric:
- Harrell C: p_adj = 0.096
- Uno C: p_adj = 1.000

M5 (DeepSurv) is significantly worse than M1 on both metrics.

### pytest Status
- **66/66 tests PASSED**
- Fixed `sa_name` attribute in test_protocol_gates.py

---

## 13. Version 1.7: Model Comparisons v4 - True Uno C Bootstrap (2026-07-24)

### Issue Identified
v3 implemented Uno C with supplementary t-test, NOT true patient-level bootstrap. The requirement was for true patient-level bootstrap for both Harrell C and Uno C.

### Key Changes from v3
- **Uno C now uses patient-level bootstrap** with IPCW from test data (as proxy for outer training cohort)
- `outer_splits_df` only contains test data, so tau estimated from test events
- Bootstrap allows some folds to fail (require at least 1 valid fold per iteration)
- Same patient sample used for both models and all 5 repeats

### Output Files
- `model_comparisons_v4.csv` - 10 comparisons (5 Harrell C + 5 Uno C)
- `model_comparisons_v4.json` - Full results with bootstrap distributions
- `AUDIT_REPORT_V3.json` - Auto-generated audit from v4 results

### Superseded Files
- `model_comparisons_v3.csv` → `model_comparisons_v3_SUPERSEDED_UNO_BOOTSTRAP_NOT_IMPLEMENTED.csv`
- `model_comparisons_v3.json` → `model_comparisons_v3_SUPERSEDED_UNO_BOOTSTRAP_NOT_IMPLEMENTED.json`
- `AUDIT_REPORT_V2.json` → `AUDIT_REPORT_V2_SUPERSEDED_INCONSISTENT_WITH_SOURCE_RESULTS.json`

### Key Results

**Harrell C-index (Patient-Level Bootstrap, 1000 iterations):**
| Comparison | Type | Diff | p_adj | Significant |
|------------|------|------|-------|-------------|
| M4 vs M1 | Formal | +0.078 | 0.096 | No |
| M3 vs M1 | Formal | +0.053 | 0.488 | No |
| M5 vs M1 | Formal | -0.079 | 0.096 | No |

**Uno C-index (Patient-Level Bootstrap, 1000 iterations):**
| Comparison | Type | Diff | p_adj | Significant |
|------------|------|------|-------|-------------|
| M4 vs M1 | Formal | +0.041 | 0.160 | No |
| M3 vs M1 | Formal | +0.019 | 1.000 | No |
| M5 vs M1 | Formal | -0.093 | **0.008** | **Yes** |

### Critical Findings

1. **M4 vs M1 is NOT significant** on either metric at Bonferroni threshold (0.0125)
   - Harrell C: p_adj = 0.096
   - Uno C: p_adj = 0.160

2. **M5 (DeepSurv) is significantly worse** than M1 on Uno C at Bonferroni threshold
   - Uno C: p_adj = 0.008 < 0.0125
   - Harrell C: p_adj = 0.096 (not significant at Bonferroni)

3. **AUDIT_REPORT_V3** correctly reflects:
   - M4 vs M1 Harrell C: p_adj = 0.0959 (NOT significant)
   - M5 vs M1 Harrell C: p_adj = 0.0959 (NOT significant)
   - M5 vs M1 Uno C: p_adj = 0.0080 (SIGNIFICANT)

---

## 14. Version 1.8: Model Comparisons v6 - Publication-Grade Statistical Closure (2026-07-24)

### Why v4 and v5 Were Superseded

- v4 estimated IPCW from outer-test data rather than the corresponding outer-training cohort and allowed an iteration to survive when only part of the 25-fold structure was evaluable.
- v5 hardcoded fold 1 in the Uno C calculation, built censoring weights on training IDs but queried them with test IDs, silently defaulted missing weights to 1.0, and omitted the finite-sample `+1` correction in the bootstrap p-value.
- Consequently, `model_comparisons_v4.*`, `model_comparisons_v5.*`, `AUDIT_REPORT_V3.json`, and `AUDIT_REPORT_V4.json` are not valid publication sources.

### v6 Locked Method

1. No model retraining and no modification of `oof_predictions.csv`.
2. Outer-training patients are reconstructed as the exact complement of each locked outer-test fold.
3. Uno C IPCW and tau are derived separately from each of the 25 outer-training folds.
4. Each bootstrap draw samples patients with replacement once and reuses that draw for both models and all five repeats.
5. Uno C is calculated separately in every fold; an iteration is valid only if all 25 fold calculations succeed.
6. The five repeat-level differences are averaged per iteration.
7. Two-sided bootstrap p-values use the finite-sample correction and therefore cannot be zero.
8. Four prespecified comparisons form the multiplicity family; Bonferroni-adjusted p-values are evaluated against family-wise alpha 0.05.

### Authoritative v6 Results

| Metric | Comparison | Mean difference | 95% CI | Raw p | Adjusted p | Significant after adjustment |
|---|---|---:|---:|---:|---:|---|
| Harrell C | M4 vs M1 | +0.0784 | [0.0170, 0.1382] | 0.0240 | 0.0959 | No |
| Uno C | M4 vs M1 | +0.0139 | [-0.0412, 0.0653] | 0.6154 | 1.0000 | No |
| Harrell C | M5 vs M1 | -0.0785 | [-0.1433, -0.0147] | 0.0240 | 0.0959 | No |
| Uno C | M5 vs M1 | -0.1303 | [-0.1861, -0.0739] | 0.0020 | 0.0080 | Yes |

### Verification

- Project Python 3.12 environment: **112 passed, 5 skipped, 0 failed**.
- All eight v6 audit gates passed.
- Locked OOF prediction SHA-256 remained `7b21074e208a563bc99b6a0a8c458f076a5ab333612e65ce2659cd9d6571228f`.
- Final status: `PHASE3A_METHOD_CLOSURE_COMPLETED`.
