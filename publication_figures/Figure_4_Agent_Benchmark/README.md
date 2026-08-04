# Figure 4 — Agent benchmark

This is the manuscript-canonical agent benchmark figure. It preserves the
frozen 4,860-record formal benchmark and incorporates the completed post-run
offline metric-definition audit. It does not rerun the language model or
replace the prespecified confirmatory endpoint.

## Canonical outputs

- `Figure_4_Agent_Benchmark.svg` — editable vector source.
- `Figure_4_Agent_Benchmark.pdf` — submission vector.
- `Figure_4_Agent_Benchmark.tiff` — 600 dpi LZW raster.
- `Figure_4_Agent_Benchmark.png` — 300 dpi preview.

## Reproduction

1. Run `python offline_scoring_audit.py`.
2. Run the five panel scripts if standalone exports are required.
3. Run `python assemble_figure4.py`.
4. Run `python verify_figure4.py`.

## Audited terminology

- `verified_task_success` → `frozen external composite pass`.
- `schema_valid` → omitted from the main figure; replaced by a post-hoc strict
  report-contract audit.
- `external_verifier_passed` → `internal deterministic verifier pass`, B4-only
  and N/A for systems without that component.
- `supported_claim_precision` → `exact extractive claim support`.
- `citation_correctness` → `retrieved-passage citation validity`.

The earlier dashboard layout is superseded and is not a manuscript source.
