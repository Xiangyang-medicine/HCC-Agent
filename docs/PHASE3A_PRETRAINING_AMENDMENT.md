# Phase 3A Pre-Training Amendment v1.1

**Date**: 2026-07-14
**Status**: PRE-TRAINING (before any model training)
**Based on**: PHASE3A_STATISTICAL_ANALYSIS_PLAN.md v1.0

---

## 1. Gender Data Extraction Results

### 1.1 Investigation Summary

Per SAP v1.1 requirement, attempted to extract gender from `cases_complete_response.json` via case_id matching.

**Findings:**

| Source | Field Path | Result |
|--------|------------|--------|
| `cases_complete_response.json` | `demographic.gender` | NOT PRESENT |
| `cases_metadata.tsv` | `gender` column | EMPTY for all 371 cases |
| GDC API direct query | `demographic.gender` | "unrecognized field" warning |
| GDC API direct query | `demographic.sex` | "unrecognized field" warning |

**Conclusion**: TCGA-LIHC demographic data does NOT include gender information in the GDC API. This is a data collection characteristic of the TCGA-LIHC cohort, not a data processing error.

### 1.2 Reported Missingness

- **Gender missing rate**: 363/363 (100%)
- **Handling**: All patients assigned `gender = "Unknown"`

### 1.3 Impact on Modeling

The SAP v1.0 specified gender as a clinical variable with:
```
gender | Categorical | 0 | Label encoding + Unknown for unseen
```

Due to 100% missingness:
- **Actual missingness**: 363/363 (100%)
- **Final model (M1, M3, M4, M5)**: Gender EXCLUDED from clinical features
- **Rationale**: 100% missing provides zero information; including would add noise

### 1.4 SAP v1.0 Correction

The following SAP v1.0 text is hereby corrected:

**Original (incorrect):**
> gender | Categorical | 0 | Label encoding + Unknown for unseen

**Corrected:**
> gender | Categorical | 363 (100%) | EXCLUDED - TCGA-LIHC does not collect gender in GDC API

---

## 2. Clinical Variables Update

### 2.1 Final Clinical Variables

| Variable | Type | Missing | Handling |
|----------|------|---------|----------|
| age_at_diagnosis | Continuous | 0 | Median imputation (in-fold) |
| ajcc_stage | Categorical | 23 (6.3%) | Ordinal encoding + Unknown for unseen |
| tumor_grade | Categorical | 5 (1.4%) | Ordinal encoding + Unknown for unseen |

**Note**: Gender excluded due to 100% missingness.

### 2.2 Model Feature Updates

| Model ID | Name | Features | Change |
|----------|------|----------|--------|
| M1 | Clinical Cox PH | age, ajcc_stage, tumor_grade | Removed gender |
| M3 | Combined Elastic-net Cox | age, stage, grade + 15 genes | Removed gender |
| M4 | Combined Random Survival Forest | age, stage, grade + 15 genes | Removed gender |
| M5 | Combined DeepSurv | age, stage, grade + 15 genes | Removed gender |

---

## 3. Clinical Category Mapping

Created `clinical_category_mapping.json` for ordinal encoding:

### 3.1 AJCC Stage Mapping

| Original | Normalized | Ordinal Value |
|----------|------------|---------------|
| Stage I, Stage IA, Stage IB | Stage I | 1 |
| Stage II, Stage IIA, Stage IIB, Stage IIC | Stage II | 2 |
| Stage III, Stage IIIA, Stage IIIB, Stage IIIC | Stage III | 3 |
| Stage IVA, Stage IVB, Stage IVC | Stage IV | 4 |
| Unknown/Missing | Unknown | 0 |

### 3.2 Tumor Grade Mapping

| Original | Ordinal Value |
|----------|---------------|
| G1 | 1 |
| G2 | 2 |
| G3 | 3 |
| G4 | 4 |
| GX | 0 |
| Unknown/Missing | 0 |

---

## 4. Smoke Test Implementation

### 4.1 Smoke Test Scope

Before full nested CV training, verify:

1. **Data Loading**: Modeling dataset loads correctly (363 patients)
2. **Preprocessing Pipeline**: In-fold scaling and encoding work
3. **Model Training**: All 5 models train on single fold
4. **Metric Calculation**: C-index computed correctly
5. **CV Split Integrity**: Each patient appears in correct folds

### 4.2 Smoke Test Acceptance Criteria

| Test | Criterion |
|------|-----------|
| Data loading | 363 patients, no missing features |
| Preprocessing | No NaN in processed features |
| M1 (Clinical) | Trains without error |
| M2 (Gene-only) | Trains without error |
| M3 (Combined) | Trains without error |
| M4 (RSF) | Trains without error |
| M5 (DeepSurv) | Trains without error |
| C-index | 0.5 < C-index < 1.0 |
| CV integrity | No patient in both train and test |

### 4.3 Smoke Test Execution

```bash
python scripts/run_phase3a_smoke_test.py
```

---

## 5. Hyperparameter Search Limits

Per SAP v1.1 amendment:

| Model | Search Limit | Configurations |
|-------|--------------|----------------|
| M2 (Gene Elastic-net) | 25 configs | 5 alpha × 5 l1_ratio |
| M3 (Combined Elastic-net) | 25 configs | 5 alpha × 5 l1_ratio |
| M4 (RSF) | 40 configs | Max 40 sampled from grid |
| M5 (DeepSurv) | 20 configs | 2 layers × 3 units × 3 dropout × 2 lr |

### 5.1 RSF Configuration Sampling

```python
# Total possible: 3 × 4 × 3 × 3 × 3 = 324 configs
# Sampled: RandomSearchCV with 40 iterations
rsf_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [3, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5]
}
```

---

## 6. DeepSurv Early Stopping Fix

### 6.1 Corrected Implementation

Early stopping validation subset must come from INNER training fold only:

```python
# WRONG (original): Use last 20% of inner training as validation
# RIGHT (corrected): Use a separate held-out subset from inner training

for inner_fold in range(n_inner_folds):
    inner_train_idx = ...  # Inner training indices
    inner_val_idx = ...    # Inner validation indices

    # Further split inner_train for early stopping
    inner_es_train, inner_es_val = train_test_split(
        inner_train_idx, test_size=0.2, random_state=seed
    )

    # Train on inner_es_train, validate on inner_es_val
    model.fit(X[inner_es_train], y[inner_es_train], ...)
```

### 6.2 IPCW Estimation

IPCW for Uno's C-index and time-dependent AUC must be estimated from OUTER training fold only:

```python
# Estimate censoring weights from outer training fold
from sksurv.metrics import concordance_index_censored

# Fit IPCW on outer_train only
ipcw = compute_ipcw(survival_train)

# Apply to outer_test predictions
cindex_uno = concordance_index_censored(
    y_test['event'], y_test['survival_months'],
    risk_scores_test,
    weighted_censor_time=ipcw
)
```

---

## 7. Model Comparison - Patient-Level Bootstrap

### 7.1 Implementation

```python
def patient_level_bootstrap(model1_scores, model2_scores, case_ids, n_iterations=1000):
    """
    Patient-level paired bootstrap for model comparison.

    Args:
        model1_scores: C-index scores for model 1 (per fold/repeat)
        model2_scores: C-index scores for model 2
        case_ids: Patient case_ids for each score
        n_iterations: Number of bootstrap iterations

    Returns:
        mean_diff, ci_lower, ci_upper, p_value
    """
    observed_diff = np.mean(model1_scores - model2_scores)

    n_patients = len(np.unique(case_ids))
    better_count = 0

    for _ in range(n_iterations):
        # Bootstrap sample at patient level
        boot_patients = np.random.choice(
            np.unique(case_ids), size=n_patients, replace=True
        )
        # Get indices for these patients
        boot_mask = np.isin(case_ids, boot_patients)
        boot_diff = np.mean(model1_scores[boot_mask] - model2_scores[boot_mask])

        if boot_diff > 0:
            better_count += 1

    p_value = 2 * min(better_count / n_iterations, 1 - better_count / n_iterations)

    return observed_diff, p_value
```

### 7.2 Bonferroni Adjustment

For 4 comparisons (M3 vs M1, M4 vs M1, M5 vs M1, M3 vs M2):
- Adjusted alpha = 0.05 / 4 = 0.0125
- Report both raw and adjusted p-values

---

## 8. Model Probability Output Requirements

Per SAP v1.1, all models must provide survival probability estimates:

### 8.1 Probability Output Specification

| Model | Method | Timepoints |
|-------|--------|------------|
| M1 (Clinical Cox) | `survfuncs` from lifelines | 12, 36, 60 months |
| M2 (Gene Elastic-net) | `survfuncs` from lifelines | 12, 36, 60 months |
| M3 (Combined Elastic-net) | `survfuncs` from lifelines | 12, 36, 60 months |
| M4 (RSF) | `predict_survival_function` | 12, 36, 60 months |
| M5 (DeepSurv) | `predict_survival_function` | 12, 36, 60 months |

### 8.2 Output Format

```python
# Per patient predictions
predictions = {
    'case_id': str,
    'repeat': int,
    'fold': int,
    'model': str,  # M1, M2, M3, M4, M5
    'risk_score': float,  # Linear predictor
    'survival_prob_12m': float,  # S(t=12)
    'survival_prob_36m': float,  # S(t=36)
    'survival_prob_60m': float,  # S(t=60)
    'survival_months': float,  # Actual survival
    'event': int  # 0=censored, 1=dead
}
```

---

## 9. Full Training Pipeline Summary

### 9.1 Execution Order

1. **Smoke Test** → Verify single fold training
2. **Full Nested CV** → 5 repeats × 5 folds × 5 models
3. **OOF Predictions** → Save all predictions
4. **Metrics Aggregation** → Compute per-fold and summary stats
5. **Model Comparison** → Bootstrap comparisons
6. **PH Assessment** → Schoenfeld residuals for Cox models
7. **Report Generation** → Final results

### 9.2 Expected Runtime

| Component | Estimated Time |
|-----------|----------------|
| Smoke Test | ~5 minutes |
| M1 (Clinical) | ~2 minutes |
| M2 (Gene Elastic-net) | ~10 minutes |
| M3 (Combined Elastic-net) | ~15 minutes |
| M4 (RSF) | ~45 minutes |
| M5 (DeepSurv) | ~2 hours |
| **Total** | ~3 hours |

---

**Amendment Author**: Claude
**Amendment Status**: READY FOR EXECUTION
**Next Step**: Execute smoke test, then full training
