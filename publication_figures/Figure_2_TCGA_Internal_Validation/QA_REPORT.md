# Figure 2 QA report

## Outcome

- Automated figure gate: **PASS, 24/24 checks**.
- Nature-figure static preflight: **PASS, 14/14 checks; 0 warnings; 0 failures**.
- Visual inspection: **PASS** after two layout iterations.

## Numerical checks

- Canonical comparison input: `model_comparisons_v6.csv`.
- Superseded v5 comparison output: not used.
- Panel a: 250 rows = 5 models × 2 metrics × 25 outer folds.
- Panel b: 8 formal comparisons.
- Panel c: 363 unique patients, each with five OOF predictions.
- Panel d: 150 rows = 2 models × 3 horizons × 25 outer folds.
- Panel e: 45 rows = 5 models × 3 analyses × 3 metrics.
- All numeric source-data cells are finite.
- M4 versus M1 remains non-significant after correction for Harrell C
  (adjusted P=0.095904) and Uno C (adjusted P=1.000).

## Visual checks

- Panel letters are distinct and aligned.
- All five panels use the same two-tier heading system: 7.6-pt bold title and
  5.35-pt muted methodological subtitle.
- Long headings were shortened and statistical metadata was moved out of the
  main title line.
- No title, subtitle, P-value column, axis label or risk-table entry is clipped.
- Panel c at-risk table is separated from panel e.
- M4 is emphasized consistently in teal without relying on color alone.
- M5 underperformance is shown rather than hidden.
- Panel e reports both actual values and ranks, including M4's SA3 Uno C rank of 3.
- Figure remains legible at a double-column width of approximately 184 mm.

## Export checks

- SVG includes editable text elements.
- PDF contains one page.
- PNG is 300 dpi.
- TIFF is 4,350 × 4,890 pixels at 600 dpi with LZW compression.
- All five standalone panels have SVG, PDF, PNG and TIFF versions.

## Final export SHA-256

- SVG: `C809E3F7A7BE0579969845C67C341ECC95C06D804F1CD44E242509D10557BE9B`
- PDF: `D945711331D0A70BC9F5AADAAEA2CA8F354836F644D5EAA9999E6940B936525A`
- PNG: `878B4048B48542311AF3A688BB88BEE71C67D8892531B023700AFF8B8F9D8A0C`
- TIFF: `5D04F95FB57655C006F7DBED0D597EECE2CF9D37C99AFDEF2D432DB2602BBD97`

## Interpretation boundary

The figure supports M4 as the provisional primary candidate with the strongest
overall internal performance profile. It does not support a claim of corrected
statistical superiority over M1, and the OOF median split is not a validated
clinical cutoff.
