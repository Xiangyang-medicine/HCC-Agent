# Table 1 quality-assurance report

Status: `TABLE1_QA_PASSED`

## Cohort reconciliation

- TCGA-LIHC: 363 patients and 129 deaths.
- GSE14520-GPL3921: 221 patients and 85 deaths.
- GSE116174-GPL570: 64 patients and 27 deaths.
- Both external cohorts match the locked Figure 3 source data.
- Age and sex are complete for every analysed patient.
- Stage counts sum to each cohort total.
- TCGA grade counts sum to 363.
- All 15 locked metabolic genes are available in all three cohorts.

## Excluded cohort

GSE14520-GPL571 contains 21 complete-OS cases and 11 events. It is recorded as
`NOT_ANALYSED` because the sample size was insufficient. It is not included as
an analysed column in Table 1.

## Workbook verification

- Formulas generate the manuscript display values from the Numeric Source tab.
- Formula-error scan matched zero cells.
- The workbook contains separate Table 1, Numeric Source, Provenance,
  Definitions and Excluded Cohort sheets.
- The rendered preview was visually inspected; headers, values and notes are
  legible without clipping or overlap.
