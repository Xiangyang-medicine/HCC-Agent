# Figure 2 revision-v2 contract

Core conclusion:
Under the frozen 100-case, three-repeat technical benchmark, the
verifier-guided closed-loop system improves externally scored end-to-end task
completion over the tool-using single-controller baseline; replanning,
verification, revision, and persistent state explain the gain.

Figure archetype:
Asymmetric mixed-modality figure with a primary-endpoint hero panel.

Target journal/output:
ACM TIST special issue; full-width two-column figure.

Backend:
Python only.

Final size:
Nominal canvas 7.25 × 6.15 inches; tight exported PDF boundary
181.6 × 152.4 mm.

Panel map:
- a: precise architectural contrast between B2 and B4.
- b: prespecified primary endpoint and paired run outcomes.
- c: paired ablation effect sizes relative to full B4.
- d: comparable extractive-traceability and repeat-stability metrics.
- e: B4-minus-B2 differences under eight frozen fault injections.

Evidence hierarchy:
- Hero evidence: panel b.
- Mechanistic support: panels a and c.
- Traceability/reliability support: panel d.
- Robustness/failure handling: panel e.

Statistics:
- Patient-clustered bootstrap 95% confidence intervals, 2,000 resamples.
- Primary B4-versus-B2 comparison: two-sided paired sign-permutation test,
  100,000 draws.
- Ablations: paired sign-permutation tests with Holm correction.
- Fault panels: patient-clustered bootstrap differences; descriptive.

Source data:
All quantitative values derive from the complete 4,860-record frozen formal
run. Post-hoc structural audit fields are explicitly labelled and do not
replace the prespecified endpoint.

Reviewer risks:
- Planning errors concern action-specification compliance, not survival labels.
- The original schema check was under-specified; it is removed from the main
  figure and replaced by a post-hoc strict structural audit.
- The internal verifier is B4-only and is not plotted as a zero-valued metric
  for systems without a verifier.
- Evidence metrics assess exact extractive support and assigned-passage
  citation validity, not expert biomedical factuality or semantic retrieval.
- Unsupported-request terminal scoring reflects a frozen scoring-contract
  mismatch and is flagged rather than silently corrected.
