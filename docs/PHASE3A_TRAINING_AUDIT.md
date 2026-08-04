# Phase 3A Training Audit Report

**Audit Date**: 2026-07-15
**Run Assessed**: Exploratory run dated 2026-07-14
**Status**: EXPLORATORY_INCOMPLETE - NOT FOR PUBLICATION
**Script Assessed**: `scripts/run_phase3a_training.py` and `scripts/run_phase3a_smoke_test.py`

---

## Executive Summary

The exploratory training run executed on 2026-07-14 failed to implement the locked nested-CV
protocol specified in `docs/PHASE3A_STATISTICAL_ANALYSIS_PLAN_v1.1.md`. This audit documents
14 specification violations that collectively invalidate all performance claims, statistical
comparisons, and deployment recommendations in the exploratory report.

Results are classified as **EXPLORATORY_INCOMPLETE_NOT_FOR_PUBLICATION**.

---

## Violation Inventory

### Violation #1: M2/M3 Not Using Inner CV for Hyperparameter Tuning

**Severity**: Critical

**Specification**: SAP v1.1 Section X requires inner CV hyperparameter tuning for M2 and M3
using `CoxnetSurvivalAnalysis` with tunable alpha and l1_ratio.

**Actual Behavior**:
```python
# run_phase3a_training.py lines 143-145, 175-177, 203-205
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)  # FIXED parameters
cph.fit(train_data, duration_col='survival_time', event_col='event')
```

**Problem**: Fixed penalizer=0.1 and l1_ratio=0.5 used for all folds without any inner CV search.
No `CoxnetSurvivalAnalysis` was used. No hyperparameter selection occurred.

**Impact**: M2 and M3 performance estimates are optimistic (no generalization via tuning).
All comparative claims involving M2 and M3 are invalid.

---

### Violation #2: No Inner CV Assignments Generated

**Severity**: Critical

**Specification**: SAP v1.1 requires generation of inner CV splits for each outer training fold
to enable proper hyperparameter selection.

**Actual Behavior**:
```python
# No inner_cv.py or inner_split generation code exists
# generate_cv_splits.py only creates outer_splits.csv
```

**Problem**: No `inner_assignments_repeat_<r>_fold_<f>.csv` files were generated.
The code directly fits models on outer training data without any inner CV structure.

**Impact**: Cannot verify that outer test data was excluded from hyperparameter selection.

---

### Violation #3: Gender Excluded from M1/M3

**Severity**: High

**Specification**: SAP v1.1 specifies gender as a clinical variable to be included in M1 and M3.

**Actual Behavior**:
```python
# run_phase3a_training.py lines 237-268
# Only uses: age_at_diagnosis, ajcc_stage, tumor_grade
features = []
features.extend([train_age, train_stage, train_grade])
```

**Problem**: Gender was explicitly excluded based on `PHASE3A_PRETRAINING_AMENDMENT.md` which
claimed 100% missingness. However, SAP v1.1 requires including gender in models regardless.

**Impact**: M1 and M3 are not implementing the specified feature set.

---

### Violation #4: Stage and Grade Using Ordinal Instead of One-Hot Encoding

**Severity**: High

**Specification**: SAP v1.1 requires one-hot encoding for stage_group, tumor_grade, and gender.

**Actual Behavior**:
```python
# run_phase3a_training.py lines 251-258
def normalize_stage(val):
    if pd.isna(val): return 0
    for v in stage_map.values():
        if val in v['original']: return v['ordinal']
    return 0
train_stage = train_df['ajcc_stage'].apply(normalize_stage).values
```

**Problem**: Stage and grade encoded as single ordinal integers (0-4 for stage, 0-4 for grade).
No one-hot encoding implemented.

**Impact**: Incorrect encoding violates SAP specification. Ordinal encoding implies incorrect
mathematical relationships between categories.

---

### Violation #5: M4 Using Fixed Parameters Instead of Nested Tuning

**Severity**: Critical

**Specification**: SAP v1.1 requires M4 Random Survival Forest with inner CV hyperparameter
selection from up to 40 candidate configurations.

**Actual Behavior**:
```python
# run_phase3a_training.py lines 222-229
rsf = RandomSurvivalForest(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
```

**Problem**: Fixed hyperparameters used. No inner CV tuning. scikit-survival was not installed
so M4 returned NaN.

**Impact**: M4 results are invalid and missing entirely.

---

### Violation #6: M5 is Placeholder Returning NaN

**Severity**: Critical

**Specification**: SAP v1.1 requires M5 DeepSurv neural network with inner CV tuning.

**Actual Behavior**:
```python
# run_phase3a_training.py lines 240-246
def train_m5_deepsurv(...):
    try:
        import torch
    except ImportError:
        log(f"  M5 skipped (PyTorch not available)", verbose=True)
        return float('nan')
    # No actual training code
    return float('nan')
```

**Problem**: M5 is completely unimplemented. No neural network training, no early stopping,
no survival probability estimation.

**Impact**: M5 is missing from all results. 1/5 models not implemented.

---

### Violation #7: Only Harrell C-index Calculated

**Severity**: High

**Specification**: SAP v1.1 requires both Harrell C-index and Uno C-index (IPCW-weighted).

**Actual Behavior**:
```python
# run_phase3a_training.py lines 314-316
from lifelines.utils import concordance_index
# concordance_index is Harrell's C-index
cidx = concordance_index(y_test, -risk_test, event_test)
```

**Problem**: Only Harrell C-index calculated. No `concordance_index_censored` or
`concordance_index_ipcw` for Uno C-index.

**Impact**: IPCW-weighted C-index not available for proper censored data analysis.

---

### Violation #8: Missing Required Metrics

**Severity**: High

**Specification**: SAP v1.1 requires Uno C-index, time-dependent AUC, Brier score, and
integrated Brier score.

**Actual Behavior**:
- No `concordance_index_ipcw` (Uno C-index)
- No `cumulative_dynamic_auc` (time-dependent AUC at 12/36/60 months)
- No `brier_score`
- No `integrated_brier_score`
- No calibration analysis

**Impact**: Incomplete evaluation. Cannot assess calibration or time-dependent discrimination.

---

### Violation #9: No Absolute Survival Probability Output

**Severity**: High

**Specification**: SAP v1.1 requires survival_probability_12m, survival_probability_36m,
survival_probability_60m for each patient.

**Actual Behavior**:
```python
# run_phase3a_training.py lines 276-287
# OOF predictions only include risk_score, survival_months, event
all_predictions.append({
    'case_id': case_id,
    'repeat': repeat,
    'fold': fold,
    'model': 'M1_clinical_cox',
    'risk_score': float(risk_test[i]),
    'survival_months': float(y_test[i]),
    'event': int(event_test[i])
})
```

**Problem**: No survival probabilities output. Only risk scores (partial hazards).

**Impact**: Cannot assess calibration or provide actionable survival estimates to clinicians.

---

### Violation #10: Invalid Statistical Comparison Using 25-Fold Paired t-test

**Severity**: High

**Specification**: SAP v1.1 requires patient-level paired bootstrap for model comparison,
not naive fold-level paired t-test.

**Actual Behavior**:
```python
# Statistical analysis executed in separate Python call
from scipy import stats
t_stat, p_val = stats.ttest_rel(m2_scores, m1_scores)
p_value = 2 * min(better_count / n_iterations, 1 - better_count / n_iterations)
```

**Problem**: Ordinary paired t-test on 25 fold-level C-index values violates independence
assumptions. Each patient appears in multiple folds across repeats, causing correlation.
Proper approach requires patient-level bootstrap.

**Impact**: P-values are unreliable. Claims of statistical significance are invalid.

---

### Violation #11: No Patient-Level Paired Bootstrap

**Severity**: High

**Specification**: SAP v1.1 requires patient-level paired bootstrap with at least 1000
iterations for model comparison.

**Actual Behavior**: No bootstrap implementation. Only ordinary t-test on fold means.

**Impact**: Cannot properly compare models. Bootstrap CI and p-values missing.

---

### Violation #12: No Sensitivity Analyses SA1/SA2/SA3

**Severity**: High

**Specification**: SAP v1.1 requires sensitivity analyses SA1 (N=363), SA2 (N=361, age>=18),
SA3 (complete cases on clinical variables).

**Actual Behavior**: None of SA1, SA2, SA3 were executed.

**Impact**: Model robustness not assessed. Publication requires sensitivity analyses.

---

### Violation #13: Invalid Conclusions Based on Inadequate Evidence

**Severity**: Critical

**Specification**: Results cannot be claimed as statistically significant, recommended for
deployment, or having clinical implications without proper protocol implementation.

**Actual Conclusions in Report** (INVALID):
- "Gene-based models significantly outperform clinical-only model" (p < 0.001)
- "Large effect" (Cohen d > 1.0)
- "Recommended for deployment"
- "Clinical implications"
- "Primary model recommendation: M2 (Gene Elastic-net)"

**Impact**: All conclusions invalidated by protocol violations. Report must not be cited.

---

### Violation #14: 21-Second Runtime Indicates Inadequate Implementation

**Severity**: Medium

**Specification**: Full nested CV with 5x5x5 fits and proper inner CV tuning should require
significantly more computation time.

**Actual Behavior**: Total runtime was 21.4 seconds for 75 model fits.

**Problem**: This runtime is only consistent with fixed-parameter fits on small data.
Proper nested CV with inner tuning, multiple hyperparameters per model, and proper
metrics would require hours.

**Impact**: Runtime confirms that proper nested CV protocol was not executed.

---

## Summary Table

| # | Violation | Severity | Impact |
|---|-----------|----------|--------|
| 1 | M2/M3 fixed params, no inner CV | Critical | Performance estimates invalid |
| 2 | No inner assignments generated | Critical | Cannot verify test isolation |
| 3 | Gender excluded from M1/M3 | High | Feature set incomplete |
| 4 | Ordinal instead of one-hot encoding | High | Incorrect encoding |
| 5 | M4 fixed params | Critical | M4 results invalid |
| 6 | M5 placeholder returning NaN | Critical | M5 not implemented |
| 7 | Harrell only, no Uno C-index | High | IPCW C-index missing |
| 8 | Missing AUC/Brier/IBS/calibration | High | Incomplete evaluation |
| 9 | No survival probabilities | High | No actionable estimates |
| 10 | Invalid 25-fold paired t-test | High | P-values unreliable |
| 11 | No patient bootstrap | High | No proper comparison |
| 12 | No SA1/SA2/SA3 | High | Robustness not assessed |
| 13 | Invalid conclusions | Critical | All claims void |
| 14 | 21-second runtime | Medium | Protocol not executed |

---

## Required Remediation

1. **Python 3.12 Environment**: Install with scikit-survival, PyTorch, pycox
2. **Formal Package**: Create `src/prognostic_engine/` with proper nested CV
3. **Inner CV Implementation**: Generate inner assignments for each outer fold
4. **One-Hot Encoding**: Implement for stage, grade, gender
5. **Hyperparameter Tuning**: Inner CV for M2, M3, M4, M5
6. **Complete Metrics**: Uno C, AUC, Brier, IBS, calibration
7. **Survival Probabilities**: Output 12/36/60 month estimates
8. **Bootstrap Comparison**: Patient-level bootstrap with 1000+ iterations
9. **Sensitivity Analyses**: Execute SA1, SA2, SA3
10. **PH Diagnostics**: Schoenfeld residuals for Cox models

---

## Files Affected

| File | Violations |
|------|------------|
| `scripts/run_phase3a_training.py` | 1,2,3,4,5,6,7,8,9,10,11,12,13,14 |
| `scripts/run_phase3a_smoke_test.py` | 1,3,4,7,9 |
| `experiments/phase3a/training/` | All output invalidated |

---

## Classification

**EXPLORATORY_INCOMPLETE_NOT_FOR_PUBLICATION**

This run must not be cited, referenced in publications, or used to support any
clinical or research conclusions. All performance numbers, statistical comparisons,
and recommendations are void and superseded by the formal protocol.

---

**Audit Conducted**: 2026-07-15
**Auditor**: Claude (automated audit)
**Files Reviewed**:
- `F:\ACM\scripts\run_phase3a_training.py`
- `F:\ACM\scripts\run_phase3a_smoke_test.py`
- `F:\ACM\experiments\phase3a\training\metrics_summary.json`
- `F:\ACM\experiments\phase3a\training\oof_predictions.csv`
