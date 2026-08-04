# Figure 4 contract

## Core conclusion

Under the frozen 100-case, three-repeat technical benchmark, the
verifier-guided closed-loop system improves externally scored end-to-end task
completion over the tool-using single-controller baseline. Replanning,
verification, revision and persistent structured state explain the gain.

## Figure specification

- Archetype: asymmetric mixed-modality figure with a primary-endpoint hero panel.
- Target: ACM TIST special issue; full-width two-column figure.
- Backend: Python.
- Canvas: 7.25 × 6.15 inches.
- Outputs: editable SVG, vector PDF, 300 dpi PNG and 600 dpi LZW TIFF.

## Panel map

- **a:** architectural contrast between B2 and B4.
- **b:** prespecified primary endpoint and paired run outcomes.
- **c:** paired component-ablation effects relative to full B4.
- **d:** extractive traceability and repeatability.
- **e:** B4-minus-B2 differences under eight frozen fault injections.

## Evidence hierarchy

- Hero evidence: panel b.
- Mechanistic support: panels a and c.
- Traceability and repeatability: panel d.
- Robustness and fault handling: panel e.

## Statistics

- Patient-clustered bootstrap 95% confidence intervals, 2,000 resamples.
- Primary B4-versus-B2 comparison: two-sided paired sign-permutation test,
  100,000 draws.
- Ablations: paired sign-permutation tests with Holm correction.
- Fault panels: patient-clustered bootstrap differences; descriptive.

## Source and audit boundary

All quantitative values derive from the complete 4,860-record frozen formal
run. Post-hoc structural audit fields are explicitly labelled and do not
replace the prespecified endpoint.

Planning errors concern action-specification compliance, not disagreement with
survival labels. Evidence metrics assess exact extractive support and
assigned-passage citation validity, not expert biomedical factuality or
semantic retrieval quality. The internal verifier is B4-only and is not
represented as a zero-valued metric for systems without that component. The
unsupported-request terminal result reflects a frozen scoring-contract
mismatch and is flagged rather than retrospectively corrected.
