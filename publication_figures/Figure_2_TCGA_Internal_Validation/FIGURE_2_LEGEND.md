# Figure 2 | Internal validation of the prognostic engine in TCGA-LIHC

**a,** Harrell and Uno concordance across all 25 outer test folds from five repeats
of five-fold nested cross-validation (N=363 patients; 129 deaths). Each point is one
outer test fold; boxes summarize the fold distribution and the thick colored segment
marks the arithmetic mean. The dashed line marks chance discrimination. M1, clinical
Cox model; M2, 15-gene elastic-net Cox model; M3, combined clinical-plus-gene
elastic-net Cox model; M4, combined random survival forest; M5, DeepSurv.

**b,** Pre-specified patient-level paired bootstrap comparisons. Points show the
mean metric difference (model A minus comparator) and horizontal lines show
percentile 95% confidence intervals from 1,000 valid bootstrap resamples. Adjusted
P values are Bonferroni-corrected within each metric family. M4 had the largest
positive mean difference from M1 for Harrell C, but this comparison did not remain
significant after correction (adjusted P=0.096); its Uno C difference from M1 was
also not significant (adjusted P=1.000).

**c,** Kaplan–Meier curves for outcome-blind patient-level OOF M4 risk. M4 risk was
ranked within each outer test fold and averaged across the five repeats, then divided
at the median (low, n=182; high, n=181). Shading shows 95% confidence intervals.
The displayed hazard ratio is from a univariable Cox model comparing high with low
OOF risk, and the P value is from a two-sided log-rank test. This grouping visualizes
internal OOF association and is not presented as an externally validated clinical
cutoff.

**d,** Brier prediction error at 12, 36 and 60 months for M1 and M4. Thin lines pair
the same outer fold, small points show all 25 folds, and large points with error bars
show the mean and 95% confidence interval across folds. Lower values indicate better
prediction. Mean integrated Brier scores were 0.196 for M1 and 0.184 for M4.

**e,** Sensitivity analyses under three cohort definitions. Cell text gives the
metric value and shading gives the within-analysis rank (1, best). SA1 is the primary
cohort (N=363); SA2 excludes patients aged <18 years (N=361); SA3 includes only
patients with complete stage and grade (N=338). M4 retained rank 1 for Harrell C and
IBS in all three analyses, while its Uno C rank was 1, 1 and 3, respectively.

All performance estimates derive from outer test folds. Uno C used fold-specific
inverse-probability-of-censoring weights estimated from the corresponding outer
training data. Source data and plotting code are provided in the panel-specific
directories.
