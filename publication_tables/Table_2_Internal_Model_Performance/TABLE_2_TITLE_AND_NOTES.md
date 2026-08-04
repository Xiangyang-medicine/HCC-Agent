# Table 2 title

**Table 2. Internal performance and prespecified paired model comparisons in TCGA-LIHC**

# Notes

Values in section A are mean (SD) across 25 outer test folds from five repeated 5-fold nested cross-validation in 363 patients. Higher values indicate better discrimination for Harrell C, Uno C, and time-dependent AUC; lower values indicate better performance for the integrated Brier score (IBS).

Section B reports the prespecified patient-level paired bootstrap comparisons. Patients were sampled with replacement, the same sampled patients were used for both models and all five repeats, the metric difference was calculated within each repeat, and repeat-specific differences were averaged in each of 1,000 valid resamples. The 95% confidence intervals are percentile intervals. Two-sided p values were Bonferroni-adjusted within each metric family of four formal comparisons. Uno C used censoring weights estimated from the corresponding outer training fold.

M4 showed the strongest descriptive performance and is retained as the provisional primary candidate; however, neither its Harrell C nor Uno C improvement over M1 was statistically significant after multiplicity correction. M5 was significantly worse than M1 for Uno C after correction.

Abbreviations: AUC, time-dependent area under the receiver operating characteristic curve; CI, confidence interval; IBS, integrated Brier score; PH, proportional hazards; RSF, random survival forest; SD, standard deviation.
