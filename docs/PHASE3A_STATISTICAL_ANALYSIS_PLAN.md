# Phase 3A Statistical Analysis Plan

**Version**: 1.0
**Date**: 2026-07-14
**Status**: PRE-LOCKED (before any model training)
**Project**: HCC Prognosis Multi-Agent LLM System

---

## 1. Study Overview

### 1.1 Research Questions

**RQ1**: Do metabolic genes provide incremental prognostic information beyond clinical variables?

**RQ2**: How do different survival models perform on internal validation in terms of discrimination and calibration?

**RQ3**: Are model results robust to age restrictions, clinical missingness, and random splits?

### 1.2 Endpoints

- **Primary endpoint**: Overall Survival (OS)
- **Event definition**: event=1 indicates death
- **Time unit**: Months
- **OS construction**: survival_months = days / 30.4375 (from Phase 2B locked definition)

### 1.3 Data Source

- **Primary dataset**: `data/processed/gdc/20260713/tcga_lihc_patients.parquet`
- **Expression data**: `data/processed/gdc/20260713/tcga_lihc_expression_tpm.parquet`
- **Expression unit for modeling**: log2(TPM + 1)

---

## 2. Input Data Specification

### 2.1 Patient Cohort

| Metric | Value |
|--------|-------|
| Total patients | 363 |
| Events (deaths) | 129 |
| Censored (alive) | 234 |
| Event rate | 35.5% |

### 2.2 Clinical Variables

| Variable | Type | Missing | Handling |
|----------|------|---------|----------|
| age_at_diagnosis | Continuous | 0 | Median imputation (in-fold) |
| gender | Categorical | 0 | Label encoding + Unknown for unseen |
| ajcc_stage | Categorical | 23 (6.3%) | Ordinal encoding + Unknown for unseen |
| tumor_grade | Categorical | 5 (1.4%) | Ordinal encoding + Unknown for unseen |

**Stage categories** (ordinal): Stage I, Stage II, Stage III, Stage IVA, Stage IVB
**Grade categories** (ordinal): G1, G2, G3, G4

### 2.3 Gene Variables (15 metabolic genes)

| Gene | Category | Ensembl ID |
|------|----------|------------|
| HK2 | Glycolysis | ENSG00000159399 |
| PKM | Glycolysis | ENSG00000067225 |
| LDHA | Glycolysis | ENSG00000134333 |
| LDHB | Glycolysis | ENSG00000111716 |
| GPI | Glycolysis | ENSG00000105220 |
| PFKL | Glycolysis | ENSG00000141959 |
| GLS | Glutamine | ENSG00000115419 |
| GLUD1 | Glutamine | ENSG00000148672 |
| FASN | Lipogenesis | ENSG00000169710 |
| SCD | Lipogenesis | ENSG00000099194 |
| CA9 | Hypoxia | ENSG00000107159 |
| VEGFA | Hypoxia | ENSG00000112715 |
| HIF1A | Hypoxia | ENSG00000100644 |
| MYC | Oncogenic | ENSG00000136997 |
| CTNNB1 | Oncogenic | ENSG00000168036 |

### 2.4 Label Leakage Prevention

- **Label columns excluded from features**: event, survival_months, vital_status, days_to_death, days_to_last_follow_up
- **Evidence**: Verified in `label_leakage_report.md`

---

## 3. Model Specifications

### 3.1 Model Set

| Model ID | Name | Features | Regularization |
|----------|------|----------|----------------|
| M1 | Clinical Cox PH | age, gender, ajcc_stage, tumor_grade | None (unpenalized) |
| M2 | Gene-only Elastic-net Cox | 15 metabolic genes | L1+L2 penalty |
| M3 | Combined Elastic-net Cox | Clinical + Genes | L1+L2 penalty |
| M4 | Combined Random Survival Forest | Clinical + Genes | Tree-based |
| M5 | Combined DeepSurv | Clinical + Genes | Neural network + early stopping |

### 3.2 Clinical Cox PH (M1)

- **Type**: Cox proportional hazards (unpenalized)
- **Features**: age (continuous), gender (binary), ajcc_stage (ordinal), tumor_grade (ordinal)
- **Parameters**: No penalty, all variables retained
- **Note**: Reference baseline for incremental value assessment

### 3.3 Gene-only Elastic-net Cox (M2)

- **Type**: Cox PH with Elastic-net regularization
- **Features**: 15 metabolic genes (log2-transformed TPM)
- **Hyperparameters to tune**:
  - alpha: [0.001, 0.01, 0.1, 0.5, 1.0]
  - l1_ratio: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Selection via inner CV**

### 3.4 Combined Elastic-net Cox (M3)

- **Type**: Cox PH with Elastic-net regularization
- **Features**: Clinical variables + 15 metabolic genes
- **Hyperparameters**: Same as M2
- **Clinical variables**: Standardized (in-fold), genes: standardized (in-fold)

### 3.5 Combined Random Survival Forest (M4)

- **Type**: Random Survival Forest
- **Features**: Same as M3
- **Hyperparameters to tune**:
  - n_estimators: [100, 200, 300]
  - max_depth: [3, 5, 7, None]
  - min_samples_split: [5, 10, 20]
  - min_samples_leaf: [3, 5, 10]
  - max_features: ['sqrt', 'log2', 0.5]
- **Selection via inner CV**

### 3.6 Combined DeepSurv (M5)

- **Type**: Deep Cox proportional hazards
- **Features**: Same as M3 (standardized)
- **Architecture constraints**:
  - 1-2 hidden layers only
  - hidden_units: [16, 32, 64]
  - dropout: [0.0, 0.1, 0.2]
  - learning_rate: [0.001, 0.01]
  - weight_decay: [0.0, 0.0001]
- **Training constraints**:
  - Random seed fixed
  - Early stopping on training loss (patience=10)
  - Max epochs: 200
  - Batch size: 32
- **Failure handling**: Log training failures, exclude failed folds from averaging

---

## 4. Internal Validation Strategy

### 4.1 Nested Cross-Validation Design

```
Outer CV (5 folds × 5 repeats) → for evaluation
    │
    └── Inner CV (5 folds) → for hyperparameter tuning within each outer train fold
```

### 4.2 Outer CV Specifications

| Parameter | Value |
|-----------|-------|
| Outer folds | 5 |
| Outer repeats | 5 |
| Total outer test sets | 25 |
| Stratification | Event rate balanced across folds |
| Seed | 42 (fixed) |

### 4.3 Patient Assignment Rules

- Each patient appears in exactly ONE outer test fold per repeat
- Each patient appears in exactly 4 outer training folds per repeat
- Same patient never appears in both train and test for same fold
- Event proportion maintained across folds (stratified)

### 4.4 Inner CV Specifications

| Parameter | Value |
|-----------|-------|
| Inner folds | 5 |
| Stratification | Event rate |
| Purpose | Hyperparameter selection |

### 4.5 Random Seeds

| Purpose | Seed |
|---------|------|
| Outer CV | 42 |
| Inner CV | 123 |
| DeepSurv | 456 |
| numpy | 42 |
| torch | 456 |

---

## 5. Preprocessing Pipeline (No Leakage)

### 5.1 Within Outer Training Fold Only

All preprocessing must be fitted on outer training fold ONLY, then applied to outer test fold:

1. **Continuous clinical (age)**: Median imputation → Standardization (mean, std)
2. **Categorical clinical (gender)**: Label encoding fitted on train → applied to test
3. **Ordinal clinical (stage, grade)**: Ordinal encoding fitted on train → applied to test
4. **Gene expression**: log2(TPM + 1) transformation → Standardization (mean, std)

### 5.2 Unseen Category Handling

- Any category in test not seen in training → mapped to "Unknown" category
- Unknown categories get index 0 (after fitting on train categories)
- Must preserve index continuity

### 5.3 Missing Value Handling

| Variable | Strategy |
|----------|----------|
| age | In-fold median imputation |
| gender | "Unknown" category |
| ajcc_stage | "Unknown" category |
| tumor_grade | "Unknown" category |

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics

| Metric | Library | Description |
|--------|---------|-------------|
| Harrell's C-index | lifelines / scikit-survival | Overall discrimination |
| Uno's C-index | scikit-survival | IPCW-weighted discrimination |

### 6.2 Secondary Metrics

| Metric | Timepoints | Note |
|--------|------------|------|
| Time-dependent AUC | 12, 36, 60 months | Only if sufficient events |
| Integrated Brier Score | 0-60 months | Only if sufficient data |
| Calibration slope | 36 months | Only if sufficient data |
| Calibration intercept | 36 months | Only if sufficient data |

### 6.3 Sufficient Data Thresholds

| Timepoint | Min Events Required | Min Patients Required |
|-----------|--------------------|-----------------------|
| 12 months | 50 | 200 |
| 36 months | 80 | 250 |
| 60 months | 100 | 300 |

If threshold not met: report `NA` with explanation, do NOT impute or fabricate.

### 6.4 IPCW Estimation

- Uno's C-index and time-dependent AUC use IPCW estimated from outer training fold
- This prevents information leakage from test fold to censoring distribution

---

## 7. Model Comparison Framework

### 7.1 Out-of-Fold Predictions

Save all outer test fold predictions to:
`experiments/phase3a/predictions/oof_predictions.parquet`

Columns:
- patient_id
- repeat
- fold
- model (M1-M5)
- risk_score
- predicted_survival_12m (if available)
- predicted_survival_36m (if available)
- predicted_survival_60m (if available)
- survival_months
- event
- fold_type (test)

### 7.2 Paired Comparison Design

Since all models share the same outer folds, use paired statistical tests:

| Comparison | Test | Adjustment |
|------------|------|------------|
| M3 vs M1 | Paired t-test | Bonferroni for 4 comparisons |
| M4 vs M1 | Paired t-test | Bonferroni |
| M5 vs M1 | Paired t-test | Bonferroni |
| M3 vs M2 | Paired t-test | Bonferroni |

### 7.3 Reporting Format

For each model (M1-M5) and each metric:
- Individual fold values (25 values)
- Mean ± SD across repeats
- 95% CI (bootstrap or analytical)
- For combined vs clinical: mean difference ± 95% CI

---

## 8. Proportional Hazards Assessment

### 8.1 Tests to Perform (Cox Models Only)

- **Global Schoenfeld residuals test**: For each repeat/fold
- **Individual variable tests**: For each clinical and gene variable
- **Time-varying coefficient test**: If global test significant

### 8.2 Reporting

| PH Violation | Action |
|--------------|--------|
| None | Report C-index normally |
| Mild (p > 0.01) | Note and report |
| Moderate (p < 0.01, p > 0.001) | Consider stratified Cox or time interaction |
| Severe (p < 0.001) | Report both PH and non-PH models |

### 8.3 Forbidden Actions

- Do NOT ignore PH violations to improve C-index
- Do NOT selectively report models without PH issues
- Any model modification due to PH must be documented

---

## 9. Sensitivity Analyses

### 9.1 Pre-specified Sensitivity Analyses

| Analysis | Dataset | N | Purpose |
|----------|---------|---|---------|
| SA1 | All patients | 363 | Primary |
| SA2 | Exclude age < 18 | 361 | Age robustness |
| SA3 | Complete clinical cases | TBD | Missingness robustness |

### 9.2 Implementation Rules

- All sensitivity analyses use same CV structure, preprocessing, and hyperparameter ranges
- SA2 excludes: TCGA-5R-AA1D (17y), TCGA-XR-A8TE (16y)
- SA3 excludes patients with missing ajcc_stage or tumor_grade
- Do NOT add post-hoc sensitivity analyses based on model results

### 9.3 CV Stability Check

Run with 3 different random seeds to verify stability:
- Primary seed: 42
- Sensitivity seed 1: 123
- Sensitivity seed 2: 2024

---

## 10. DeepSurv Specific Requirements

### 10.1 Training Protocol

```
For each outer fold × repeat × inner fold:
  1. Preprocess (in-fold)
  2. Initialize network with seed
  3. Train with early stopping
  4. Save training curve
  5. Evaluate on inner test
```

### 10.2 Failure Criteria

| Failure Mode | Handling |
|--------------|----------|
| Training divergence | Stop, log as failure, exclude from averaging |
| NaN predictions | Stop, log as failure, exclude from averaging |
| Not converging (>200 epochs) | Use best checkpoint, log warning |
| GPU memory error | Retry with smaller batch, log as failure |

### 10.3 Reporting

- Total training attempts
- Successful trainings
- Failed trainings (with reason)
- Per-fold success rate

---

## 11. Model Selection Rules

### 11.1 Selection Criteria (Pre-specified)

Final model selection considers ALL of the following:

| Criterion | Weight | Measurement |
|-----------|--------|------------|
| Uno C-index | High | Mean across outer folds |
| IBS | High | Mean across outer folds |
| Calibration | Medium | Slope, intercept |
| Stability | Medium | SD across repeats |
| Complexity | Low | Number of non-zero coefficients |
| PH assumption | High | Schoenfeld p-value |
| Probability output | Medium | Availability of reliable survival probabilities |

### 11.2 Candidate Models (max 3)

- **Primary candidate**: Best combined model (M3 or M4 or M5) per criteria above
- **Simplified clinical baseline**: M1 (Clinical Cox PH)
- **Alternative model**: If primary candidate has issues, select alternative

### 11.3 Forbidden Selection Behaviors

- Do NOT select model based on single best fold
- Do NOT cherry-pick folds to include/exclude
- Do NOT report only best-performing sensitivity analysis
- Do NOT modify model after seeing results

---

## 12. Output Directory Structure

```
experiments/phase3a/<run_id>/
├── config.json              # Run configuration
├── environment.json          # Environment specs
├── package_versions.txt      # Python package versions
├── input_hashes.json         # SHA-256 of input data
├── splits/
│   ├── outer_splits.csv      # Patient-folds assignments
│   └── inner_split_config.json # Inner CV config
├── models/
│   ├── M1_clinical_cox/      # One folder per model
│   ├── M2_gene_elasticnet/
│   ├── M3_combined_elasticnet/
│   ├── M4_combined_rsf/
│   └── M5_combined_deepsurv/
├── predictions/
│   └── oof_predictions.parquet # All out-of-fold predictions
├── metrics/
│   ├── cindex_uno.csv        # Uno C-index per fold/repeat
│   ├── cindex_harrell.csv    # Harrell C-index
│   ├── auc_time_dependent.csv # Time-dependent AUC
│   ├── brier_score.csv       # IBS
│   └── calibration.csv       # Calibration metrics
├── diagnostics/
│   ├── ph_assessment.csv     # PH test results
│   ├── deepsurv_failures.json
│   └── preprocessing_log.csv
├── logs/
│   └── training_logs/        # DeepSurv training curves
└── README.md                 # Run documentation
```

---

## 13. Forbidden Actions During Phase 3A

1. **LLM Agent development**: Do not develop or call LLM agents
2. **Mock agents**: Do not use mock agents for evaluation
3. **External validation**: Do not begin external validation (ICGC, GEO)
4. **Parameter tuning on test data**: Do not use test fold for any decisions
5. **Outcome-based data selection**: Do not exclude patients based on outcomes
6. **Feature selection on full data**: All feature selection in-fold only
7. **Result-driven analysis**: Do not add analyses after seeing results
8. **Claiming external validity**: Do not attribute internal metrics to external populations

---

## 14. Success Criteria for Phase 3A Completion

| Criterion | Requirement |
|-----------|------------|
| OS audit match | 363/363 PASS |
| All models trained | M1-M5 complete |
| Nested CV completed | 25 outer folds × 5 models |
| OOF predictions saved | Complete with no missing patients |
| Metrics calculated | All planned metrics with proper NA handling |
| PH assessment | All Cox models assessed |
| Sensitivity analyses | SA1, SA2, SA3 complete |
| No data leakage | Verified by unit tests |

---

## 15. Deliverables

Upon Phase 3A completion, report:

1. OS audit match rate after hotfix
2. Age outlier reporting accuracy
3. Final modeling N and event count
4. All nested CV results for all models
5. Incremental value: Clinical vs Combined models
6. Calibration and IBS for each model
7. PH assumption check results
8. DeepSurv training failures or instabilities
9. Sensitivity analysis results
10. All fold failures (if any)
11. Candidate model selection with pre-defined rationale
12. Input data and code hashes
13. Unit test results
14. Remaining work (external validation, LLM layer)

---

**Plan Locked**: 2026-07-14
**Plan Author**: Claude (independent verification)
**Next Phase**: Phase 3B (External Validation, after Phase 3A completes)
**Phase 4**: LLM Multi-Agent Layer (after Phase 3B)
