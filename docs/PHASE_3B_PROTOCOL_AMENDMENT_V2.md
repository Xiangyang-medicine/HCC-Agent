# Phase 3B Protocol Amendment v2 - External Validation of M4

**Date:** 2026-07-27  
**Status:** FROZEN BEFORE EXTERNAL DATA ACQUISITION OR ANALYSIS  
**Amends:** `PHASE_3B_VALIDATION_PROTOCOL.md` v1.0  
**Reason:** Phase 3A selected M4 (combined Random Survival Forest) as the provisional primary candidate. M4 was not significantly superior to M1 after multiplicity adjustment; external validation is confirmatory for transportability, not a vehicle to establish superiority.

## 1. Primary Model and Scope

- **Primary external-validation model:** M4 combined RSF (clinical variables plus the prespecified 15-gene feature panel).
- **Clinical reference:** M1 clinical Cox.
- **Secondary models:** M2 and M3, reported only when the required features are available.
- **M5 DeepSurv:** excluded from external primary analysis because it was inferior to M1 on internal Uno C after multiplicity correction.
- **No claim of clinical utility, treatment selection, or deployment readiness is permitted.**

## 2. Locked Derivation Procedure

Before any external cohort file is opened for analysis:

1. Refit M4 once using all 363 verified TCGA-LIHC derivation cases.
2. Select M4 hyperparameters only through the prespecified inner cross-validation procedure on the TCGA derivation set.
3. Freeze the model artifact, feature order, gene mapping, missing-data policy, software environment, random seeds, and SHA-256 hashes.
4. Record all chosen hyperparameters and the full derivation manifest.
5. Do not retrain, recalibrate, choose features, alter cut-points, or tune hyperparameters using any external outcome or performance result.

The external model artifact must be designated `EXTERNAL_VALIDATION_ONLY`; it is not a clinical deployment artifact.

## 3. Cohorts and Analysis Hierarchy

| Role | Cohort | Technology | Analysis status |
|---|---|---|---|
| Primary | ICGC LIRI-JP, if complete RNA-seq and OS data are obtainable | RNA-seq | Confirmatory external validation |
| Secondary | A second independent RNA-seq HCC cohort, if feature-compatible | RNA-seq | Confirmatory only if specified before download and eligibility is met |
| Exploratory | GEO GSE14520 | Microarray | Exploratory cross-platform transportability analysis only |

GSE14520 must not be combined with RNA-seq cohorts and cannot rescue a failed RNA-seq validation claim. Its endpoint definition (OS versus recurrence-free survival), gene/probe coverage, and platform transformation must be reported before any model application.

## 4. Eligibility and Feature Compatibility

Required per case for M4:

- unambiguous patient identifier;
- overall-survival time and event indicator;
- all clinical predictors required by the frozen M4 artifact;
- all 15 prespecified gene features after the locked identifier mapping.

Do not impute missing external model features using outcome-informed methods or external-cohort distribution tuning. The primary M4 complete-case analysis will exclude feature-incomplete cases and report the exclusion flow. A prespecified training-derived imputation sensitivity analysis may be added only after it is implemented and tested before external analysis.

## 5. Preprocessing and Harmonization

- Map identifiers using a versioned mapping table frozen before scoring.
- Apply the exact derivation feature order.
- Fit transformations and imputation parameters on the TCGA derivation data only.
- Do not use ComBat, quantile normalization, or any joint source-plus-external transformation in the confirmatory analysis.
- For microarray data, collapse probes by the prespecified mapping rule and report every unavailable gene. If direct-scale transport is not technically defensible, report no M4 primary score rather than inventing a cross-platform calibration.

## 6. Endpoints and Statistical Analysis

Primary external endpoint: Uno C-index of the frozen M4 risk score for overall survival, reported with a 95% confidence interval using the locked survival-metric implementation.

Secondary endpoints:

- Harrell C-index;
- time-dependent AUC at only estimable, prespecified horizons;
- integrated Brier score and calibration, only when sample size and follow-up support estimation;
- M4 versus M1 paired comparison within the same external cases;
- transparent reporting of not-estimable metrics rather than substitutions.

For each cohort, bootstrap patients with replacement, preserve all model pairings, use 1000 iterations, and report confidence intervals, valid/invalid iteration counts, and the exact code/data hash. Comparisons across cohorts are descriptive unless an explicit, prespecified meta-analytic procedure is added before analysis.

## 7. Decision and Reporting Rules

- Report point estimates and uncertainty; do not use arbitrary C-index thresholds to label a model “validated.”
- The primary claim is limited to performance in the named external cohort under the frozen pipeline.
- Any recalibration is a separate, exploratory analysis and must be clearly separated from frozen-model validation.
- A negative or inconclusive result is reported without model replacement or post hoc cohort switching.
- All downloaded source files, source URLs, dates, checksums, sample flow, feature mapping, and exclusions must be archived.

## 8. Required Gates Before Scoring

1. Source-data provenance gate passed.
2. Cohort eligibility and outcome definition audited.
3. Frozen TCGA-derived M4 artifact and manifest present.
4. Feature compatibility report present before outcome performance is calculated.
5. No external outcome or score is consumed before all preceding gates pass.
6. Unit tests for gene ordering, missing-feature abstention, model hash, and metric inputs pass.

## 9. Relationship to Phase 4

Phase 3B evaluates the deterministic prognostic tool. It does not validate the LLM system. Phase 4 agent evaluation must continue to keep model output, evidence synthesis, verification, and fault recovery analytically separate.
