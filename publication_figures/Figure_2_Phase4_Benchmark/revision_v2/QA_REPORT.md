# Figure 2 revision-v2 quality-assurance report

## Outcome

Status: `FIGURE2_V2_QA_PASSED`

The final figure was rendered with the Python backend only and visually
inspected at full exported resolution. No clipping, unreadable labels,
panel collisions, or color-only categorical distinctions were found.

## Methodological audit

- Frozen formal records inspected: 4,860/4,860.
- Raw formal records modified: no.
- Confirmatory endpoint replaced: no.
- Post-hoc strict report-contract disagreements in B2–B4 clean runs: 0/600.
- Planning errors are action-specification failures, not disagreement with
  survival outcomes or clinical labels.
- Evidence endpoints are automated exact-extractive and assigned-passage
  checks, not expert biomedical factuality labels.

## Figure integrity

- Required panels: a–e present.
- Quantitative panel source data: present.
- Main-figure `schema_valid` metric: absent.
- Inapplicable zero-valued external-verifier comparison: absent.
- Numeric source data contain no infinite values.
- Primary paired counts sum to 300.
- Primary B4-minus-B2 difference: 13.0 percentage points.

## Export checks

- SVG: present; text remains editable.
- PDF: one page; exported boundary 181.6 × 152.4 mm.
- PNG: 300 dpi.
- TIFF: 600 dpi, LZW compression, minimum-dimension check passed.
- Font family: Arial/Helvetica/sans-serif fallback.
- Minimum explicitly configured global type size: 6.6 pt; panel labels 8.5 pt.
- Static preflight: 14 PASS, 0 WARN, 0 FAIL.
- Figure-specific QA: 25/25 checks passed.

## Visual encoding

- B2 and B4 are distinguished by both direct labels and color.
- Fault-handling endpoints are distinguished by marker shape and color.
- No rainbow palette is used.
- Confidence intervals and the paired-outcome matrix are directly annotated.

## Interpretation boundary

The figure supports a technical benchmark claim: verifier-guided closed-loop
execution improves the frozen externally scored composite task-pass rate and
failure handling relative to the tool-using single-controller baseline.
It does not establish clinical utility, superior survival prediction, expert
medical factuality, deployment readiness, treatment benefit, or patient
outcome improvement.
