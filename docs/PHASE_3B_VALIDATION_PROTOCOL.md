# Phase 3B External Validation Protocol

**Version**: 1.0
**Date**: 2026-07-15
**Status**: PRE-REGISTERED
**Prerequisite**: Phase 3A must pass integrity gates

---

## 1. Validation Objective

External validation of the Phase 3A prognostic models (M1, M2, M3) on independent cohorts to assess:

1. **Generalizability**: Performance on data from different sources
2. **Transportability**: Performance in different populations
3. **Calibration drift**: Whether predictions remain accurate over time/population

---

## 2. External Datasets

### 2.1 ICGC-LIRI-JP (Japanese Liver Cancer Cohort)
- **Source**: ICGC Data Portal (dcc.icgc.org)
- **Accession**: LIRI-JP
- **Expected N**: ~200 patients with complete clinical + expression data
- **Population**: Japanese (vs. TCGA's multi-ethnic US population)
- **Platform**: Illumina HiSeq (harmonized with TCGA)

**Required Variables**:
- Survival time (OS in months)
- Event indicator (death)
- Clinical covariates (stage, grade, age, sex)
- RNA-seq expression for 15 metabolic genes

### 2.2 GEO GSE14520 (Chinese HCC Cohort)
- **Source**: Gene Expression Omnibus (GEO)
- **Accession**: GSE14520
- **Expected N**: ~247 patients
- **Population**: Chinese
- **Platform**: Affymetrix GPL3921 (requires cross-platform normalization)

**Required Variables**:
- Survival time (RFS or OS)
- Event indicator
- Clinical covariates
- Gene expression (normalized)

---

## 3. Preprocessing Protocol

### 3.1 ICGC-LIRI-JP
```
1. Download harmonized expression via ICGC API
2. Map gene symbols to ENSEMBL IDs
3. Extract 15 metabolic gene expressions
4. Merge with clinical data
5. Apply same exclusion criteria as Phase 3A:
   - Exclude: <18 years, non-HCC diagnosis, missing survival
6. Log2 transform if counts (already normalized)
```

### 3.2 GEO GSE14520
```
1. Download raw CEL files or normalized matrix
2. Apply RMA normalization if raw
3. Map probe IDs to gene symbols (collapse by max)
4. Apply ComBat or remove batch effects if needed
5. Standardize to TCGA distribution (quantile normalization)
6. Match clinical variables to TCGA schema
```

---

## 4. Model Application Protocol

### 4.1 Feature Standardization
- Apply same preprocessing as Phase 3A:
  - Clinical: StandardScaler fitted on TCGA training data
  - Genes: StandardScaler fitted on TCGA training data

### 4.2 Prediction Pipeline
```
For each external cohort:
    1. Load Phase 3A fitted scalers
    2. Standardize external data using TCGA-fitted scalers
    3. Apply trained models (M1, M2, M3) to standardized features
    4. Generate risk scores and survival probabilities
    5. Compute validation metrics
```

### 4.3 Handling Missing Data
- If gene expression missing: Impute using cohort median
- If clinical variable missing: Exclude from M1/M3 (not M2)
- Document imputation in results

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics (Same as Phase 3A)
- Harrell C-index (with 95% CI via bootstrap)
- Uno C-index (IPCW-weighted)
- Time-dependent AUC at 12/36/60 months
- Integrated Brier Score

### 5.2 Calibration Assessment
- Calibration slope and intercept
- Calibration plots at 36 months
- Observed vs. Expected ratio

### 5.3 Performance Degradation Metrics
- Absolute C-index drop from Phase 3A
- Relative performance (external / internal)
- Categorical: Acceptable (<0.05 drop), Marginal (0.05-0.10), Poor (>0.10)

---

## 6. Sample Size Requirements

### 6.1 Minimum Sample Size
Based on acceptable precision for C-index estimation:
- **Minimum**: N = 100 (for 95% CI width of ~0.10)
- **Target**: N = 200 (for 95% CI width of ~0.07)

### 6.2 Event Requirements
- Minimum 50 events for stable calibration
- Target event rate: 30-50%

---

## 7. Statistical Analysis

### 7.1 Confidence Intervals
- Bootstrap with 1000 iterations
- Report 95% CI for all metrics
- Compare to Phase 3A using Wald test for C-index

### 7.2 Decision Rules
```
IF C-index_drop < 0.05 AND Uno_C > 0.60:
    STATUS = "VALIDATED"
    CLAIM = "Performance generalizes to external cohorts"

ELIF C-index_drop < 0.10 AND Uno_C > 0.55:
    STATUS = "CONDITIONALLY_VALIDATED"
    CLAIM = "Performance shows partial transportability"

ELSE:
    STATUS = "VALIDATION_FAILED"
    CLAIM = "Model requires recalibration for external populations"
```

---

## 8. Reporting Requirements

### 8.1 Required Tables
1. Cohort characteristics (Table 1: demographics)
2. Performance metrics by cohort (Table 2)
3. Calibration metrics (Table 3)
4. Performance comparison TCGA vs. external (Table 4)

### 8.2 Required Figures
1. Calibration plots at 36 months (one per cohort, one pooled)
2. Time-dependent AUC curves
3. Kaplan-Meier by risk groups (low/medium/high)

---

## 9. Forbidden Claims

Until Phase 3B is completed:

| Forbidden Claim | Reason |
|----------------|--------|
| "M3 generalizes to other populations" | Requires external validation |
| "The model is ready for clinical use" | External validation is necessary |
| "Performance is equivalent across ethnicities" | Requires formal comparison |

---

## 10. Success Criteria

**Phase 3B is considered SUCCESSFUL if**:

1. **Data acquisition**: Both ICGC-LIRI-JP and GEO GSE14520 successfully obtained
2. **Preprocessing**: All 15 metabolic genes extracted with <20% missingness
3. **Primary endpoint**: At least one cohort achieves:
   - C-index >= 0.58 (adjusted for expected ~0.05 drop)
   - Uno C >= 0.55
   - IBS <= 0.20

4. **Transparency**: All deviations from protocol documented

---

## 11. Timeline

| Week | Task |
|------|------|
| 1-2 | Data acquisition (ICGC, GEO) |
| 3-4 | Preprocessing and QC |
| 5 | Model application |
| 6 | Statistical analysis |
| 7 | Report generation |

---

*This protocol is pre-registered and binding. Deviations require documented justification.*
