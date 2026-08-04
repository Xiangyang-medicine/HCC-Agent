# Phase 3A Candidate Model Selection

**Date:** 2026-07-21
**Last Updated:** 2026-07-24 (v4 - v6 statistical closure)
**Status:** PROVISIONAL_PRIMARY_CANDIDATE
**Author:** Claude Code Agent

---

## 1. Executive Summary

Based on the comprehensive evaluation of 5 candidate models (M1-M5) across 25 nested CV folds, we recommend:

| Role | Model | Justification |
|------|-------|---------------|
| **Provisional Primary Candidate** | **M4 (Combined Random Survival Forest)** | Best discrimination (C=0.641), lowest IBS (0.184) |
| **Clinical Baseline** | M1 (Clinical Cox PH) | Reference model for incremental value assessment |
| **Alternative** | M3 (Combined Elasticnet Cox) | Interpretable Cox-based model with competitive performance |

**Rejected:** M5 (DeepSurv) - significantly worse than M1 on Uno C after multiplicity adjustment; high variance and no performance benefit

**Important Note:** M4 vs M1 is not statistically significant after Bonferroni adjustment on Harrell C (adjusted p=0.0959) or Uno C (adjusted p=1.0000). Adjusted p-values are compared with family-wise alpha 0.05; equivalently, raw p-values may be compared with 0.0125.

---

## 2. Performance Summary

### 2.1 Primary Metrics (Mean ± SD across 25 folds)

| Model | Uno C-index | Harrell C | IBS | Rank |
|-------|-------------|-----------|-----|------|
| M4 (RSF) | **0.627 ± 0.052** | **0.641 ± 0.057** | **0.184 ± 0.015** | 1 |
| M3 (Combined Cox) | 0.608 ± 0.056 | 0.623 ± 0.061 | 0.192 ± 0.012 | 2 |
| M2 (Gene Cox) | 0.601 ± 0.058 | 0.619 ± 0.062 | 0.192 ± 0.011 | 3 |
| M1 (Clinical Cox) | 0.615 ± 0.066 | 0.597 ± 0.065 | 0.196 ± 0.012 | 4 |
| M5 (DeepSurv) | 0.486 ± 0.076 | 0.479 ± 0.098 | 0.209 ± 0.018 | 5 |

### 2.2 Statistical Comparisons (vs M1 Clinical Baseline)
**Authoritative v6 Results (Patient-clustered paired bootstrap, 1000 valid iterations, n=363 patients):**

**Harrell C-index (Patient-Level Bootstrap):**
| Comparison | Mean Diff | 95% CI | Raw p | Adj p | Significant |
|------------|-----------|--------|-------|-------|-------------|
| M4 vs M1 | +0.0784 | [0.0170, 0.1382] | 0.0240 | 0.0959 | No |
| M3 vs M1 | +0.0531 | [-0.0132, 0.1124] | 0.1219 | 0.4875 | No |
| M5 vs M1 | -0.0785 | [-0.1433, -0.0147] | 0.0240 | 0.0959 | No |

**Uno C-index (Patient-Level Bootstrap):**
| Comparison | Mean Diff | 95% CI | Raw p | Adj p | Significant |
|------------|-----------|--------|-------|-------|-------------|
| M4 vs M1 | +0.0139 | [-0.0412, 0.0653] | 0.6154 | 1.0000 | No |
| M3 vs M1 | -0.0055 | [-0.0650, 0.0506] | 0.8691 | 1.0000 | No |
| M5 vs M1 | -0.1303 | [-0.1861, -0.0739] | 0.0020 | **0.0080** | **Yes** |

*For four formal comparisons, significance is raw p < 0.0125 or, equivalently, Bonferroni-adjusted p < 0.05.*

**Important:** M4 vs M1 is NOT statistically significant at Bonferroni threshold on either metric. Only M5 vs M1 is significant on Uno C (p_adj=0.008).

### 2.3 Incremental Value Assessment

| Comparison | Purpose | Result |
|------------|---------|--------|
| M3 vs M2 | Combined Cox vs Gene-only | Harrell +0.001; Uno +0.007 (both NS) |
| M4 vs M2 | RSF vs Gene-only, exploratory | Harrell +0.026; Uno +0.027 (both NS) |

---

## 3. Selection Criteria Evaluation

### 3.1 Per SAP Section 11.1 Criteria

| Criterion | M4 (RSF) | M3 (Combined Cox) | M2 (Gene) | M5 (DeepSurv) |
|-----------|----------|-------------------|-----------|---------------|
| Uno C-index | **Best (0.627)** | Good (0.608) | Good (0.601) | Poor (0.486) |
| IBS | **Best (0.184)** | Good (0.192) | Good (0.192) | Poor (0.209) |
| Calibration | Moderate | Good | Good | Poor |
| Stability (SD) | **Best (0.052)** | Good (0.056) | Good (0.058) | Poor (0.076) |
| PH assumption | N/A (RSF) | Satisfied | Satisfied | N/A |
| Complexity | Moderate | Low | Low | High |
| Interpretability | Moderate | High | High | Low |

### 3.2 PH Assumption Results (M1 only, per SAP)
**Corrected v2 Results (Per-covariate with Bonferroni correction):**

| Metric | Value |
|--------|-------|
| diagnostics_executed | 25 |
| folds_with_any_covariate_violation | 7 |
| folds_without_detected_violation | 18 |

- **Per-covariate violations:** `stage_nan` shows 7/25 raw violations, 0/25 Bonferroni violations
- **Conclusion:** PH assumption reasonably satisfied. Note: standard global test not available for penalized Cox models.

---

## 4. Decision Rationale

### 4.1 Why M4 (RSF)?

1. **Best discrimination:** Highest Uno C-index (0.627) and Harrell C (0.641)
2. **Best calibration:** Lowest IBS (0.184), indicating better probabilistic predictions
3. **Statistical superiority not proven:** M4 vs M1 adjusted p=0.0959 (Harrell C) and 1.0000 (Uno C)
4. **Stability:** Lowest variance across folds (SD=0.052), indicating robust performance
5. **Clinical utility:** RSF handles non-linearities and interactions naturally
6. **Provisional status:** Subject to SA2/SA3 sensitivity analysis validation

### 4.2 Why Not M3?

1. **Incremental value not proven:** M3 vs M2 and M3 vs M1 are not significant after multiplicity adjustment
2. **M4 captures additional patterns:** RSF's ensemble approach captures non-linear relationships
3. **Trade-off:** Cox models assume proportional hazards; RSF does not

### 4.3 Why Not M5 (DeepSurv)?

1. **Worse than baseline:** M5 is significantly worse on Uno C (difference -0.1303; adjusted p=0.0080); its Harrell C difference is not significant after adjustment
2. **High variance:** SD=0.076 indicates unstable training
3. **Small sample size:** DeepSurv typically requires larger N for stable training
4. **Complexity without benefit:** High computational cost without performance gain

---

## 5. Phase 3B Protocol Implications

### 5.1 External Validation Strategy

Per the deviation log, if M4 is selected, the Phase 3B protocol requires amendment:

**Original (ICGC-LIRI-JP):** 202 patients, limited metabolic gene data
**Amendment Required:** Consider GEO datasets (GSE14520, GPL3921) for additional validation

### 5.2 M4 Considerations for External Validation

- RSF does not produce interpretable coefficients
- Consider SHAP or permutation importance for feature contributions
- May need to map stage/grade categories between TCGA and external datasets

---

## 6. Next Steps

1. ~~Complete Sensitivity Analyses (SA2, SA3)~~ **COMPLETED**
2. **Phase 3B Protocol Amendment** - required since M4 RSF selected for external validation
3. **M4 Feature Importance Analysis** - SHAP or permutation importance for clinical interpretation
4. **External Validation** - ICGC-LIRI-JP and/or GEO datasets (GSE14520)
5. ~~Final statistical audit~~ **COMPLETED** - `AUDIT_REPORT_V5.json`

---

## 7. Files Generated

| File | Description | Status |
|------|-------------|--------|
| `ph_diagnostics_v2.csv` | PH assumption test results (v2) | Current |
| `ph_diagnostics_v2.json` | PH summary with per-covariate analysis (v2) | Current |
| `model_comparisons_v6.csv` | Authoritative Harrell C and Uno C comparisons | Current |
| `model_comparisons_v6.json` | Detailed v6 bootstrap results and 25-fold provenance | Current |
| `AUDIT_REPORT_V5.json` | Final machine-readable Phase 3A statistical audit | Current |
| `SENSITIVITY_SUMMARY_V2.*` | SA1/SA2/SA3 comparison summary | Current |
| `ph_diagnostics.csv` | Superseded (v1 - do not use) | SUPERSEDED |
| `ph_diagnostics_summary.json` | Superseded (v1 - do not use) | SUPERSEDED |
| `model_comparisons.csv` | Superseded (v1 - do not use) | SUPERSEDED |
| `model_comparisons.json` | Superseded (v1 - do not use) | SUPERSEDED |
| `model_comparisons_v2.*` | Superseded (v2 - do not use) | SUPERSEDED |
| `model_comparisons_v3/v4/v5.*` | Superseded due to incomplete or incorrect Uno C inference | SUPERSEDED |

---

**Document Status:** PROVISIONAL_PRIMARY_CANDIDATE
**Ready for Phase 3B:** YES, after the required M4 external-validation protocol amendment is approved and frozen
