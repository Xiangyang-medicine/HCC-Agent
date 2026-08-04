# Phase 4 post-run metric clarification and offline scoring audit

## Status

This is a post-run audit. It does not modify the 4,860 frozen formal records,
their hashes, or the prespecified confirmatory endpoint.

## Planning errors

A planning error is disagreement with the frozen action specification, not
disagreement with a patient outcome or survival label. The required plan is
`prognostic_tool -> evidence_tool` with exact `case_id`, `repeat`, and
`requested_model` arguments.

- B4 Verifier-guided closed loop: 41/300 initially invalid; 34 repaired; 7 finally invalid.
- B2 Tool-using single controller: 33/300 initially invalid; 0 repaired; 33 finally invalid.

B4 repaired 34
initially invalid plans through its single allowed replanning step.

## Metric corrections

- `schema_valid` is not used in the revised main figure. The audit instead
  reports `strict_report_contract_complete`, which validates the typed model
  payload, 12/36/60-month probabilities, source hash, provenance, claims,
  citation-ID lists, and non-empty report text.
- `external_verifier_passed` is renamed
  `internal_deterministic_verifier_pass` and is N/A for systems without that
  component.
- `supported_claim_precision` is renamed
  `exact_extractive_claim_support`.
- `citation_correctness` is renamed
  `retrieved_passage_citation_validity`.
- `verified_task_success` is reported as
  `frozen_external_composite_pass`.

## Sensitivity result

The stricter structural audit changed 0 of
1500 clean-system records. The B4-versus-B2 confirmatory estimate
therefore remains 13.0 percentage points
(95% CI 8.0 to 18.3);
the strict audit is nevertheless labelled post hoc.

## Interpretation boundary

Evidence metrics assess exact extractive support and passage-ID validity under
the frozen assigned-passage contract. They are not expert-annotated clinical
factuality, semantic retrieval quality, clinical utility, or patient benefit.
