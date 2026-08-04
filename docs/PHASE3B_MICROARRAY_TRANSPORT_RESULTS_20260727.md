# Phase 3B secondary microarray transport analysis — results

**Status:** completed, exploratory secondary analysis  
**Integrity gate:** `experiments/phase3b/microarray_transport/MICROARRAY_TRANSPORT_GATE.json` (`success: true`)  
**Frozen artifact:** `M2T_15gene_crossplatform_v1`, SHA-256 `eb4973d193d3a63af27ebcb71d85d2df092214f03ea81cce37eefda8988a0ab0`

## Claim boundary

This analysis evaluates a new TCGA-derived, gene-only elastic-net Cox transport model (`M2T`) in two independently sourced HCC microarray series. It is not an external validation of the RNA-seq-plus-clinical M4 random-survival-forest model and does not establish clinical utility. The pre-existing legacy GSE figures, scores, scripts, and result files were not imported.

The M2T model was fit once using the verified 363-case/129-event TCGA derivation cohort. Its 15 genes and hyperparameters (alpha 0.10, `l1_ratio` 0.70) were selected using TCGA-only inner cross-validation. For each microarray platform stratum, official GEO expression and platform annotation data were mapped using the locked median probe-collapse rule, then standardised without access to outcome data before the frozen artifact generated one score per patient.

## Main-text cohort-specific performance

| Cohort-platform stratum | N / events | Harrell C (95% bootstrap CI) | Uno C (95% bootstrap CI) | Interpretation |
|---|---:|---:|---:|---|
| GSE116174 GPL570 | 64 / 27 | 0.603 (0.492–0.711) | 0.601 (0.483–0.712) | directionally favourable but imprecise |
| GSE14520 GPL3921 | 221 / 85 | 0.629 (0.571–0.687) | 0.635 (0.574–0.696) | favourable secondary cross-platform evidence |

The separate GSE14520 GPL571 platform stratum (21 cases/11 events; Harrell C 0.480 and Uno C 0.447) is not included in the main-text table because it is too small for a stable summary estimate. It remains fully reported in the Supplement and in `GSE14520_GPL571_M2T_EVALUATION.json`; see `PHASE3B_MICROARRAY_REPORTING_DEVIATION_20260727.md`.

Each interval used 1,000 patient-level bootstrap draws. Patients are not pooled across cohorts or platforms. The time horizon for Uno C was the prespecified 90th percentile of observed event time in each evaluated stratum; it was reported in the corresponding evaluation JSON.

## Publication wording

Permitted wording: *“A TCGA-derived 15-gene transport model showed exploratory cross-platform discrimination in an independent 221-patient HCC microarray stratum (Uno C 0.635, 95% CI 0.574–0.696) and directionally consistent but imprecise performance in a 64-patient cohort. A 21-patient platform stratum is reported descriptively in the Supplement because its sample size precluded a stable main-text estimate.”*

Not permitted: “M4 was externally validated,” “the agent was externally validated,” “the signature is clinically validated,” or any pooled headline C-index that hides the GPL571 result.

## Remaining publication-critical work

1. Obtain at least one eligible RNA-seq cohort with OS and the full M4 clinical feature set for the primary M4 external-validation question; the microarray results cannot substitute for it.
2. Integrate these modality-labelled deterministic results into the Phase 4 agent benchmark, with the required `EXPLORATORY_CROSS_PLATFORM_RESEARCH_ONLY` limitation in generated reports.
3. Prepare manuscript figures/tables directly from the canonical JSON/CSV outputs and include full source, mapping, exclusion, hash, and legacy-exclusion provenance in the supplement.
