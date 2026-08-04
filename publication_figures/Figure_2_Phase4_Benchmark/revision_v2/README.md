# Figure 2 revision v2

This revision preserves the frozen 4,860-record formal benchmark and adds a
post-run, offline metric-definition audit. It does not rerun the language model
or replace the prespecified confirmatory endpoint.

## Canonical outputs

- `Figure_2_Phase4_Benchmark_v2.svg` — editable vector source.
- `Figure_2_Phase4_Benchmark_v2.pdf` — submission vector.
- `Figure_2_Phase4_Benchmark_v2.tiff` — 600 dpi LZW raster.
- `Figure_2_Phase4_Benchmark_v2.png` — 300 dpi preview.

## Reproduction

1. Run `python offline_scoring_audit.py`.
2. Run each panel script if standalone panel exports are required.
3. Run `python assemble_figure2_v2.py`.
4. Run `python verify_figure2_v2.py`.

## Terminology corrections

- `verified_task_success` → `frozen external composite pass`.
- `schema_valid` → omitted from the main figure; replaced by a post-hoc strict
  report-contract audit.
- `external_verifier_passed` → `internal deterministic verifier pass`, B4-only
  and N/A for systems without that component.
- `supported_claim_precision` → `exact extractive claim support`.
- `citation_correctness` → `retrieved-passage citation validity`.

The previous dashboard layout remains available in the parent directory but is
superseded for manuscript use by this revision.

