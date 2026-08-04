# Table 1 — cohort characteristics

This folder contains the manuscript Table 1 package.

## Final outputs

- `Table_1_Cohort_Characteristics.xlsx` — formatted, editable workbook.
- `Table_1_Cohort_Characteristics.csv` — machine-readable manuscript table.
- `TABLE_1_TITLE_AND_NOTES.md` — manuscript title and notes.

## Reproducibility

1. Run `python prepare_table1_source.py`.
2. Run `build_table1.mjs` with the bundled spreadsheet runtime.
3. Confirm the source and workbook gates and visually inspect the rendered
   Table 1 preview.

The `source_data` folder contains the numeric cohort summary, provenance with
SHA-256 hashes, the excluded-cohort record and the workbook payload.
