# Figure 3 contract

Core conclusion:
A TCGA-derived, outcome-blind metabolic-gene transport model showed
exploratory cross-platform survival discrimination in two independent HCC
microarray cohorts, with stronger and more precise evidence in the larger
GSE14520 GPL3921 stratum.

Figure archetype:
Asymmetric quantitative grid with two complementary survival-effect panels.

Target journal/output:
ACM TIST special issue; full-width two-column figure.

Backend:
Python only.

Final size:
Nominal canvas 7.25 × 5.85 inches.

Panel map:
- a: external Kaplan–Meier survival separation using a frozen TCGA-derived
  median risk threshold.
- b: continuous transported-score association with OS (HR per 1 SD).
- c: frozen penalized coefficient profile for the 15-gene input panel.
- d: cohort-specific Harrell and Uno concordance with bootstrap intervals.

Evidence hierarchy:
- Hero evidence: panel d.
- External patient-level survival separation: panel a.
- Complementary continuous-effect evidence: panel b.
- Model-content transparency: panel c.

Statistics:
- Harrell C and Uno C reported separately for each cohort.
- Patient-level bootstrap 95% confidence intervals, 1,000 draws.
- Continuous-score Cox HR per 1 SD with model-based 95% CI.
- Frozen-threshold risk-group HRs, log-rank tests and numbers at risk.
- No patient pooling and no pooled headline estimate.

Source data:
Canonical Phase 3B M2T artifact, manifests, passed transport gate,
patient-level evaluation records and cohort-specific evaluation JSON files.

Reviewer risks:
- This is secondary exploratory transport analysis, not external validation of
  M4.
- The 15-gene input panel has two non-zero penalized coefficients.
- External platform standardization was outcome blind but cohort specific.
- The risk-group threshold was the median score in the locked TCGA derivation
  cohort and was applied unchanged to both external cohorts.
- The panel-b Cox coefficient is an evaluation effect estimate; it was not
  used to recalibrate the frozen risk score.
- No external outcome-driven cut-point, prognostic-model coefficient fitting,
  recalibration or hyperparameter tuning was used.
- GPL571 (N=21) was not analysed because of insufficient precision.
