# Phase 3B Protocol Amendment v3 — Secondary Microarray Transportability Analysis

**Date:** 2026-07-27  
**Status:** FROZEN FOR CANONICAL REANALYSIS  
**Amends:** `PHASE_3B_PROTOCOL_AMENDMENT_V2.md`  
**Transparency status:** post-access, pre-canonical-reanalysis. The project directory contains legacy analyses of GSE14520 and GSE116174. Their scores, figures, tables, code, model weights, and conclusions are excluded. This amendment was written before the new canonical pipeline produces any risk score or evaluates any external outcome.

## 1. Purpose and claim boundary

This amendment adds two independent HCC microarray cohorts with overall-survival information:

| Cohort | Intended platform | Role |
|---|---|---|
| GSE14520 | GPL571 and GPL3921 | secondary, cross-platform transportability cohort |
| GSE116174 | GPL570 | secondary, cross-platform transportability cohort |

They **do not validate the frozen M4 combined RSF directly**. M4 requires clinical variables and `log2(TPM + 1)` RNA-seq inputs, so direct application to microarray values would be invalid. They instead evaluate a new, TCGA-derived, gene-only transport model named `M2T_15gene_crossplatform`.

The permitted manuscript claim is limited to: *the prespecified 15-gene prognostic component was evaluated without external outcome-informed fitting in two independent microarray cohorts under an explicitly exploratory, cross-platform transport protocol.* No result may be described as confirmation of M4, clinical utility, individual risk prediction, or treatment selection.

## 2. Locked M2T derivation procedure

1. Use only the verified 363-case TCGA-LIHC derivation dataset and its prespecified 15 genes.
2. Fit a gene-only elastic-net Cox model. Select the penalty and `l1_ratio` only with the existing TCGA-only five-fold inner CV procedure; all standardisation is fit in the relevant TCGA training data.
3. Refit exactly once on all 363 TCGA cases after selecting those TCGA-only hyperparameters.
4. Freeze the selected feature order, coefficients, intercept/baseline information where applicable, preprocessing specification, source-code hash, input-data hash, random seeds, and artifact hash before any external score is generated.
5. The artifact is labelled `EXPLORATORY_EXTERNAL_TRANSPORT_ONLY_NOT_FOR_CLINICAL_DEPLOYMENT`.

## 3. Prespecified microarray transformation

For each cohort-platform stratum independently:

1. Use the official GEO series-matrix expression file and official platform annotation file, both retained with URL, download date, and SHA-256 hash.
2. Restrict to tumour samples with a unique clinical match and complete OS time/event data. The expression transformation does not read time or event values.
3. Map probes to HGNC symbols using the versioned GEO platform annotation. Probes mapping to more than one of the 15 symbols are excluded. For a gene with multiple eligible probes, take the arithmetic median of the available log2-normalised probe values for each sample.
4. Require all 15 genes. Any platform stratum that lacks a required gene is not scored and is reported as `FEATURE_INCOMPATIBLE`.
5. Standardise each of the 15 collapsed gene values using the mean and sample standard deviation of the **eligible external cohort-platform stratum**, without outcome access. The same feature order is used by the frozen TCGA-derived M2T coefficients. This is unsupervised scale harmonisation, not recalibration.
6. Do not apply ComBat, quantile normalisation across TCGA and external data, outcome-informed probe selection, external feature selection, coefficient refitting, recalibration, cut-point selection, or hyperparameter tuning.

This is a distinct transport model rather than a direct-scale validation of the existing RNA-seq model. It is consequently secondary and exploratory.

## 4. External outcome analysis

Only after the source, sample-match, probe-map, feature-completeness, and frozen-artifact gates pass:

- calculate one continuous risk score per eligible patient;
- report Harrell C-index and Uno C-index separately for every cohort-platform stratum;
- use patient-level bootstrap (1,000 draws) for 95% confidence intervals; report valid and invalid draws;
- use OS time and event exactly as specified by each source, retaining a field-level derivation audit;
- do not derive a median cut-point or report dichotomised Kaplan–Meier curves as a primary test;
- do not pool individual patients across cohorts. A two-cohort summary, if presented, is descriptive and displays cohort-specific estimates first.

Any metric that cannot be estimated is reported as `NOT_ESTIMABLE`; no fallback metric or endpoint substitution is permitted.

## 5. Mandatory gates and failure policy

Before scoring, the canonical pipeline must pass all of the following:

1. Official source URL, date, local hashes, and platform metadata are recorded.
2. Each retained patient has a unique expression-to-clinical mapping.
3. The OS time/event coding and missing-data flow are written to an audit file.
4. All 15 genes have passed the fixed probe-mapping rule.
5. The TCGA-only M2T artifact is present and its hash matches the manifest.
6. Legacy GSE result files and scripts are not imported by the canonical code path.
7. Unit tests cover probe collapse, missing-gene abstention, outcome-blind scoring, patient uniqueness, and artifact/hash checks.

Failure of any gate prohibits score generation for that stratum. A null or unfavourable performance result will be retained and reported; no replacement cohort or model may be selected after observing it.

## 6. Relationship to the paper and Phase 4

The RNA-seq external-validation question for M4 remains open and is not replaced by this analysis. The microarray analysis strengthens evidence of gene-layer robustness only. It also supplies a modality-labelled deterministic tool result for Phase 4: the agent must identify it as `EXPLORATORY_CROSS_PLATFORM_RESEARCH_ONLY` and must not translate it into patient-specific clinical advice.
