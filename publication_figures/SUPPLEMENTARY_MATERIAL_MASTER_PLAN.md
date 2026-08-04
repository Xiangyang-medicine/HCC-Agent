# Supplementary material master plan

Status: implemented and ready for unified submission review.

## Governing rule

- Use a **figure** only when position, shape, trend, density, flow, or a
  continuous relationship carries the scientific message.
- Use a **table** when the primary content is an exact value, checklist,
  per-fold record, mapping inventory, threshold, or audit trail.
- The rejected table-like diagnostic montage is archived and is not for
  submission.

## Supplementary figures

### Supplementary Figure S1 — Time-dependent discrimination and calibration

- **a:** Fold-level AUC trajectories at 12, 36 and 60 months for M1–M5.
- **b–d:** Repeat-aware OOF calibration curves for M1 and M4 at 12, 36 and
  60 months, using Kaplan–Meier observed event probabilities within each
  repeat.

Exact calibration and missingness values are reported in Supplementary Table
S2.

### Supplementary Figure S2 — Repeated OOF prediction stability

- **a:** Pairwise repeat-to-repeat Spearman correlations.
- **b:** Patient-level SD of five within-repeat risk percentiles.
- **c–d:** Joint-density displays for M4 and M5 repeat-pair risk percentiles.
- **e:** Modal-quintile consensus distribution and structural QC.

### Supplementary Figure S3 — Cross-platform input and mapping quality

- **a:** Cohort flow, including GPL571 only as screened/excluded at N=21.
- **b:** Eligible probe coverage for all 15 genes on the included platforms.
- **c–d:** Expression-space PCA before and after outcome-blind within-cohort
  standardisation.
- **e:** Frozen gene-component score distributions in the two included
  cohorts.

This figure assesses input compatibility for exploratory transport of the
frozen gene-only component, not external validation of the full M4 model.

### Supplementary Figure S4 — Agent benchmark reliability and recovery

- **a:** Formal-case composition by OOF risk-quintile and observed-event
  sampling stratum.
- **b:** Paired case-level B2 versus B4 task-success fractions.
- **c:** Three-repeat reliability and exact agreement.
- **d:** B4 initial-plan to terminal-state verification/repair flow.
- **e:** Fault-injection terminal-outcome composition.
- **f:** Exact claim-support precision versus citation completeness against
  the frozen offline reference.

No clinical-utility or physician-performance claim is permitted.

## Supplementary tables

### Supplementary Table S1 — Data provenance

Canonical sources, checksums, cohorts and analysis boundaries.

### Supplementary Table S2 — Complete internal validation metrics

All 25-fold discrimination, prediction-error, calibration, below-chance,
missingness and corrected proportional-hazards diagnostics. This table absorbs
the exact-number content of the rejected old S1.

### Supplementary Table S3 — Model specification and training details

Feature sets, algorithms, nested-CV design, seeds, software versions, all 125
model–fold selected settings, failure handling and code/source hashes.

### Supplementary Table S4 — External cohort mapping inventory

Included cohorts, endpoints, exact probe-to-gene mapping for all 15 genes,
locked mapping rules and source hashes. GPL571 appears only as excluded:
N=21, insufficient sample size, no performance analysis.

### Supplementary Table S5 — Agent benchmark definitions

System variants, 100-case taxonomy, corrected metric names, deterministic
verifier rules, eight fault definitions and the frozen evidence-reference
inventory.

### Supplementary Table S6 — Complete Agent benchmark and audit results

Clean and repeat summaries, paired primary comparison, ablations, fault
effects, planning audit, post-hoc audit and all 4,860 formal run records.

## Cross-material reporting boundaries

1. M4 remains a provisional primary candidate; superiority over M1 is not
   statistically established.
2. GSE14520 GPL3921 and GSE116174 GPL570 are secondary exploratory
   cross-platform transport cohorts for the frozen gene-only component.
3. GPL571 has no reported performance result.
4. Agent evaluation establishes technical benchmark performance only, not
   clinical utility.
