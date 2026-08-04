# Figure 2 contract

Core conclusion:
In the frozen 100-case, three-repeat Phase 4 benchmark, the full closed-loop
multi-agent system (B4) improves verified task success over the strongest
tool-using single-agent baseline (B2), with stronger verification, grounding,
and fault handling, at the cost of additional latency and token use.

Figure archetype:
Quantitative grid with a dominant primary-endpoint panel.

Target journal/output:
ACM TIST special issue; full-width two-column figure.

Backend:
Python (matplotlib/seaborn only).

Final size:
7.25 x 9.20 inches.

Panel map:
- a: Prespecified primary endpoint: verified task success for B0-B4.
- b: Clean-case functional endpoint decomposition for B1-B4.
- c: Evidence grounding and unsupported-claim rates for B1-B4.
- d: B4 component ablations with paired, multiplicity-adjusted comparisons.
- e: Three-repeat agreement versus latency, with token use encoded by point size.
- f: Failure detection and correct recovery/safe-abstention matrices.

Evidence hierarchy:
- Hero evidence: panel a, B4 versus B2.
- Validation evidence: panels b and c.
- Controls/robustness: panels d and e.
- Cost/reliability trade-off: panel f.

Statistics:
- Rates and differences use patient-clustered bootstrap 95% confidence
  intervals (2,000 resamples).
- The sole confirmatory comparison is B4 versus B2 for verified task success,
  using a two-sided paired sign-permutation test (100,000 draws).
- B4 ablations use paired sign-permutation tests with Holm correction.
- Other panels are prespecified descriptive/secondary analyses.

Source data:
All plotted values are derived from the 4,860-record formal Phase 4 output.
Each panel folder contains its own source-data CSV and plotting script.

Reviewer risks:
- B0 is a non-agent quantitative reference and cannot satisfy the composite
  agent endpoint by design.
- Repeated runs are not treated as independent patients; inference clusters by
  case_id.
- Test-retest agreement is reported as exact three-run agreement because
  prevalence makes kappa unstable for near-constant endpoints.
- Fault outcomes must be interpreted per the frozen scoring contract; high
  failure detection does not imply successful recovery for every fault.
- No panel supports clinical utility, treatment guidance, deployment readiness,
  or improved patient outcomes.
