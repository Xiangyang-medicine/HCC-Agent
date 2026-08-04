# Figure 2 legend — revision v2

**Fig. 2 | Verifier-guided closed-loop execution improves externally scored
task completion and failure handling.** **a,** Architectural contrast between
the tool-using single-controller baseline (B2) and the proposed
verifier-guided closed-loop system (B4). Both systems used the same language
model backend, frozen prognostic-model tool, assigned evidence passages and
formal cases. B4 additionally used a deterministic verifier and conditional
replanning, tool retry and one synthesis revision. **b,** Prespecified frozen
external composite-pass rate across 100 formal cases and three repeated runs
per case. Points show patient-clustered means and error bars show bootstrap 95%
confidence intervals (2,000 resamples). B4 passed 284/300 runs (94.7%) versus
245/300 (81.7%) for B2, an absolute difference of 13.0 percentage points (95%
CI, 8.0–18.3; two-sided paired sign-permutation test, p=2.0×10⁻⁵; 100,000
draws). The inset reports paired run-level outcomes. A post-hoc typed structural
audit changed 0/600 B2–B4 clean runs and did not replace the prespecified
endpoint. **c,** Paired change in the composite-pass rate after removal of each
B4 component. Points and error bars show patient-clustered mean differences and
bootstrap 95% confidence intervals. Displayed p values are two-sided paired
sign-permutation tests with Holm correction across four ablations. **d,**
Comparable traceability and repeat-stability endpoints for B2 and B4. Exact
extractive support requires a generated claim to be an exact sentence from an
assigned cited passage; citation validity requires every citation identifier
to belong to the assigned passage set. These automated endpoints are not expert
biomedical factuality labels. Exact three-run agreement is the proportion of
cases with the same binary composite outcome in all three runs. **e,**
Patient-clustered B4-minus-B2 differences in failure detection and correct
terminal outcomes for eight frozen fault types (30 cases × three repeats per
system and fault). For the unsupported-request fault, B4 detected the forbidden
request and safely exited, but the frozen terminal-outcome rule required task
completion and therefore scored the exit as unsuccessful. The figure reports
this scoring-contract mismatch without post-hoc correction. All quantitative
results derive from the complete 4,860-record frozen formal output. No panel
assesses clinical utility, diagnostic accuracy, treatment benefit, deployment
readiness or patient outcomes.

