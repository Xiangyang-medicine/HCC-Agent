# Figure 2 legend

**Fig. 2 | Closed-loop agent benchmark performance.** **a,** Verified task
success on 100 clean formal cases, each evaluated in three repeated runs. Bars
show mean success and patient-clustered bootstrap 95% confidence intervals
(2,000 resamples). B4 achieved 94.7% (284/300) versus 81.7% (245/300) for the
strongest prespecified tool-using single-agent baseline B2, an absolute
difference of 13.0 percentage points (95% CI, 8.0–18.3; two-sided paired
sign-permutation test, *p*=2.0×10⁻⁵; 100,000 permutations). B0 is a non-agent
quantitative reference and cannot satisfy the composite agent endpoint by
design. **b,** Decomposition of clean-case planning, tool selection, schema,
numeric-fidelity, and verifier endpoints for B1–B4. **c,** Supported-claim
precision, citation correctness, and unsupported-claim rate under the frozen
extractive evidence contract. **d,** B4 component ablations. Error bars are
patient-clustered bootstrap 95% confidence intervals; displayed *p* values are
two-sided paired sign-permutation tests versus full B4 with Holm correction
across four ablations. **e,** Exact agreement of the binary verified-success
endpoint across all three runs versus median wall latency for B2–B4. Error bars
are patient-clustered bootstrap 95% confidence intervals; point area scales
with mean total token use. Exact agreement is reported because endpoint
prevalence makes chance-corrected kappa unstable. **f,** Failure detection and
prespecified correct recovery or safe-abstention rates for eight injected fault
types (30 cases × three repeats per cell). Fault detection does not necessarily
imply a correct terminal outcome. All analyses use the complete 4,860-record
formal dataset without case exclusion. Source data are provided in the
corresponding panel folders.
