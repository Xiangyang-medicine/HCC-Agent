import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.TABLE3_ROOT || SCRIPT_DIR;
const SOURCE_DIR = path.join(ROOT, "source_data");
const OUTPUT = path.join(ROOT, "Table_3_External_Transport.xlsx");
const PREVIEW = path.join(ROOT, "Table_3_Preview.png");

const payload = JSON.parse(
  await fs.readFile(path.join(SOURCE_DIR, "table3_payload.json"), "utf8"),
);

const workbook = Workbook.create();
const table = workbook.worksheets.add("Table 3");
const performance = workbook.worksheets.add("Performance Source");
const threshold = workbook.worksheets.add("Threshold Source");
const coefficients = workbook.worksheets.add("Frozen Coefficients");
const excluded = workbook.worksheets.add("Excluded Cohort");
const provenance = workbook.worksheets.add("Provenance");
const definitions = workbook.worksheets.add("Definitions");

const navy = "#24364B";
const blue = "#536C8B";
const orange = "#C65D3A";
const paleBlue = "#E8EEF4";
const paleOrange = "#F6E8E2";
const paleGray = "#F4F5F7";
const midGray = "#66707C";
const rule = "#C7CDD4";
const text = "#20252B";

for (const sheet of [
  table,
  performance,
  threshold,
  coefficients,
  excluded,
  provenance,
  definitions,
]) {
  sheet.showGridLines = false;
}

table.mergeCells("A1:I1");
table.getRange("A1").values = [[payload.table_title]];
table.mergeCells("A2:I2");
table.getRange("A2").values = [[payload.subtitle]];
table.getRange("A1:I1").format = {
  fill: navy,
  font: { name: "Arial", size: 14, bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
table.getRange("A2:I2").format = {
  fill: paleBlue,
  font: { name: "Arial", size: 9, italic: true, color: midGray },
  verticalAlignment: "center",
};
table.getRange("A1:I1").format.rowHeight = 28;
table.getRange("A2:I2").format.rowHeight = 22;

table.mergeCells("A4:I4");
table.getRange("A4").values = [[
  "A. Discrimination and continuous transported-score association",
]];
table.getRange("A4:I4").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};

table.getRange("A5:I5").values = [[
  "External cohort",
  "Patients",
  "Deaths, n (%)",
  "Harrell C\n(95% CI)",
  "Uno C\n(95% CI)",
  "Uno τ,\nmonths",
  "HR per 1-SD score\n(95% CI)",
  "Wald p",
  "PH-test p",
]];
table.getRange("A5:I5").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    bottom: { style: "medium", color: navy },
  },
};
table.getRange("A5").format.horizontalAlignment = "left";
table.getRange("A5:I5").format.rowHeight = 38;

table.getRange("A6:I7").values = payload.performance.map((row) => [
  `${row.cohort} (${row.platform})`,
  row.n,
  "",
  "",
  "",
  row.uno_tau_months,
  "",
  "",
  row.continuous_ph_test_p,
]);
table.getRange("A6:I7").format = {
  font: { name: "Arial", size: 9, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
table.getRange("B6:I7").format.horizontalAlignment = "center";
table.getRange("A6:A7").format.horizontalAlignment = "left";
table.getRange("A6:I6").format.fill = paleBlue;
table.getRange("A7:I7").format.fill = paleOrange;
table.getRange("A6:I7").format.rowHeight = 36;

const perfHeaders = [
  "Cohort",
  "Platform",
  "N",
  "Events",
  "Event rate",
  "Median observed time, months",
  "Harrell C",
  "Harrell CI lower",
  "Harrell CI upper",
  "Uno C",
  "Uno CI lower",
  "Uno CI upper",
  "Uno tau, months",
  "Bootstrap draws",
  "Valid iterations",
  "Continuous HR per 1 SD",
  "Continuous CI lower",
  "Continuous CI upper",
  "Continuous Wald p",
  "PH-test p",
  "Risk standardisation",
  "Cutpoint used",
  "External recalibration",
];
performance.getRange("A1:W1").values = [perfHeaders];
const perfRows = payload.performance.map((row) => [
  row.cohort,
  row.platform,
  row.n,
  row.events,
  row.event_rate,
  row.median_observed_time_months,
  row.harrell_c,
  row.harrell_ci_lower,
  row.harrell_ci_upper,
  row.uno_c,
  row.uno_ci_lower,
  row.uno_ci_upper,
  row.uno_tau_months,
  row.n_bootstrap,
  row.valid_iterations,
  row.continuous_hr_per_1sd,
  row.continuous_ci_lower,
  row.continuous_ci_upper,
  row.continuous_wald_p,
  row.continuous_ph_test_p,
  row.risk_standardisation,
  row.cutpoint_used_for_continuous_effect,
  row.external_recalibration,
]);
performance.getRange("A2:W3").values = perfRows;
performance.getRange("A1:W3").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
performance.getRange("A1:W1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
performance.getRange("C2:T3").format.numberFormat = "0.000000";
performance.getRange("A1:B5").format.columnWidth = 18;
performance.getRange("C1:W5").format.columnWidth = 16;
performance.freezePanes.freezeRows(1);
performance.tables.add("A1:W3", true, "Table3PerformanceSource");

const nPctFormula = (sourceRow) =>
  `=TEXT('Performance Source'!D${sourceRow},"0")&" ("&TEXT('Performance Source'!E${sourceRow},"0.0%")&")"`;
const estimateCiFormula = (estimateCol, lowerCol, upperCol, sourceRow, digits) =>
  `=TEXT('Performance Source'!${estimateCol}${sourceRow},"0.${"0".repeat(digits)}")&" ("&TEXT('Performance Source'!${lowerCol}${sourceRow},"0.${"0".repeat(digits)}")&"–"&TEXT('Performance Source'!${upperCol}${sourceRow},"0.${"0".repeat(digits)}")&")"`;
const pFormula = (sourceCol, sourceRow) =>
  `=IF('Performance Source'!${sourceCol}${sourceRow}<0.001,"<0.001",TEXT('Performance Source'!${sourceCol}${sourceRow},"0.000"))`;
for (let index = 0; index < 2; index += 1) {
  const targetRow = index + 6;
  const sourceRow = index + 2;
  table.getRange(`C${targetRow}`).formulas = [[nPctFormula(sourceRow)]];
  table.getRange(`D${targetRow}`).formulas = [[
    estimateCiFormula("G", "H", "I", sourceRow, 3),
  ]];
  table.getRange(`E${targetRow}`).formulas = [[
    estimateCiFormula("J", "K", "L", sourceRow, 3),
  ]];
  table.getRange(`G${targetRow}`).formulas = [[
    estimateCiFormula("P", "Q", "R", sourceRow, 2),
  ]];
  table.getRange(`H${targetRow}`).formulas = [[pFormula("S", sourceRow)]];
}
table.getRange("F6:F7").format.numberFormat = "0.0";
table.getRange("I6:I7").format.numberFormat = "0.000";

table.mergeCells("A9:I9");
table.getRange("A9").values = [[
  "B. Frozen TCGA-threshold survival stratification",
]];
table.getRange("A9:I9").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};
table.getRange("A10:I10").values = [[
  "External cohort",
  "Higher-risk n",
  "Lower-risk n",
  "Higher vs lower risk HR\n(95% CI)",
  "Log-rank p",
  "Frozen cutoff",
  "Cutoff origin",
  "External outcome used\nfor grouping",
  "External recalibration",
]];
table.getRange("A10:I10").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    bottom: { style: "medium", color: navy },
  },
};
table.getRange("A10").format.horizontalAlignment = "left";
table.getRange("G10").format.horizontalAlignment = "left";
table.getRange("A10:I10").format.rowHeight = 40;

table.getRange("A11:I12").values = payload.threshold.map((row) => [
  `${row.cohort} (${row.platform})`,
  row.higher_risk_n,
  row.lower_risk_n,
  "",
  "",
  row.frozen_tcga_cutoff,
  row.cutoff_origin,
  row.external_outcome_used_for_grouping ? "Yes" : "No",
  "No",
]);
table.getRange("A11:I12").format = {
  font: { name: "Arial", size: 9, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
table.getRange("B11:F12").format.horizontalAlignment = "center";
table.getRange("H11:I12").format.horizontalAlignment = "center";
table.getRange("A11:A12").format.horizontalAlignment = "left";
table.getRange("G11:G12").format.horizontalAlignment = "left";
table.getRange("A11:I11").format.fill = paleBlue;
table.getRange("A12:I12").format.fill = paleOrange;
table.getRange("A11:I12").format.rowHeight = 36;
table.getRange("F11:F12").format.numberFormat = "0.0000";

const thresholdHeaders = [
  "Cohort",
  "Platform",
  "Higher-risk n",
  "Lower-risk n",
  "Hazard ratio",
  "CI lower",
  "CI upper",
  "Cox Wald p",
  "Log-rank p",
  "Frozen TCGA cutoff",
  "Cutoff origin",
  "External outcome used for grouping",
];
threshold.getRange("A1:L1").values = [thresholdHeaders];
const thresholdRows = payload.threshold.map((row) => [
  row.cohort,
  row.platform,
  row.higher_risk_n,
  row.lower_risk_n,
  row.hazard_ratio,
  row.ci_lower,
  row.ci_upper,
  row.cox_wald_p,
  row.logrank_p,
  row.frozen_tcga_cutoff,
  row.cutoff_origin,
  row.external_outcome_used_for_grouping,
]);
threshold.getRange("A2:L3").values = thresholdRows;
threshold.getRange("A1:L3").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
threshold.getRange("A1:L1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
threshold.getRange("C2:J3").format.numberFormat = "0.000000";
threshold.getRange("A1:L5").format.columnWidth = 21;
threshold.freezePanes.freezeRows(1);
threshold.tables.add("A1:L3", true, "Table3ThresholdSource");

const thresholdHrFormula = (sourceRow) =>
  `=TEXT('Threshold Source'!E${sourceRow},"0.00")&" ("&TEXT('Threshold Source'!F${sourceRow},"0.00")&"–"&TEXT('Threshold Source'!G${sourceRow},"0.00")&")"`;
const thresholdPFormula = (sourceRow) =>
  `=IF('Threshold Source'!I${sourceRow}<0.001,"<0.001",TEXT('Threshold Source'!I${sourceRow},"0.000"))`;
for (let index = 0; index < 2; index += 1) {
  const targetRow = index + 11;
  const sourceRow = index + 2;
  table.getRange(`D${targetRow}`).formulas = [[thresholdHrFormula(sourceRow)]];
  table.getRange(`E${targetRow}`).formulas = [[thresholdPFormula(sourceRow)]];
}

const notes = [
  "Harrell C and Uno C use patient-level bootstrap 95% percentile CIs from 1,000 valid resamples; cohorts were evaluated separately and were not pooled.",
  "The continuous HR is per 1-SD higher transported score; PH-test p values used ranked-time transformation. No cutoff was used for this analysis.",
  "The categorical cutoff (−0.0100) was the median score in locked TCGA derivation data and was applied unchanged to both external cohorts.",
  "External outcomes were not used for mapping, standardisation, coefficient fitting, tuning, cutoff selection, or recalibration.",
  "GSE14520 GPL571 (N=21, 11 deaths) was not analysed because the sample size was insufficient for a stable main analysis.",
  "These results evaluate cross-platform transport of the frozen gene-only component; they are not external validation of M4, clinical utility, or deployment readiness.",
];
for (let index = 0; index < notes.length; index += 1) {
  const row = index + 14;
  table.mergeCells(`A${row}:I${row}`);
  table.getRange(`A${row}`).values = [[notes[index]]];
}
table.getRange("A14:I19").format = {
  font: { name: "Arial", size: 8, color: midGray },
  wrapText: true,
  verticalAlignment: "center",
};
table.getRange("A14:I19").format.rowHeight = 24;
table.getRange("A14:I14").format.borders = {
  top: { style: "medium", color: navy },
};

table.getRange("A1:A22").format.columnWidth = 27;
table.getRange("B1:C22").format.columnWidth = 15;
table.getRange("D1:E22").format.columnWidth = 24;
table.getRange("F1:F22").format.columnWidth = 14;
table.getRange("G1:G22").format.columnWidth = 27;
table.getRange("H1:I22").format.columnWidth = 22;
table.freezePanes.freezeRows(5);

coefficients.getRange("A1:E1").values = [[
  "Feature order",
  "Gene",
  "Frozen coefficient",
  "Non-zero",
  "Direction",
]];
const coefficientRows = payload.coefficients.map((row) => [
  Number(row.feature_order),
  row.gene,
  Number(row.coefficient),
  String(row.nonzero).toLowerCase() === "true",
  row.direction,
]);
coefficients.getRange("A2:E16").values = coefficientRows;
coefficients.getRange("A1:E16").format = {
  font: { name: "Arial", size: 9, color: text },
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
coefficients.getRange("A1:E1").format = {
  fill: navy,
  font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
};
coefficients.getRange("C2:C16").format.numberFormat = "0.000000";
coefficients.getRange("A1:E20").format.columnWidth = 23;
coefficients.freezePanes.freezeRows(1);
coefficients.tables.add("A1:E16", true, "Table3FrozenCoefficients");

excluded.getRange("A1:F1").values = [[
  "Cohort",
  "Platform",
  "Complete OS cases",
  "Deaths",
  "Analysis status",
  "Reason",
]];
excluded.getRange("A2:F2").values = [[
  payload.excluded.cohort,
  payload.excluded.platform,
  payload.excluded.complete_os_cases,
  payload.excluded.events,
  payload.excluded.analysis_status,
  payload.excluded.reason,
]];
excluded.getRange("A1:F2").format = {
  font: { name: "Arial", size: 9, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
excluded.getRange("A1:F1").format = {
  fill: navy,
  font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
};
excluded.getRange("A1:E4").format.columnWidth = 21;
excluded.getRange("F1:F4").format.columnWidth = 58;

provenance.getRange("A1:D1").values = [[
  "Input file",
  "Role",
  "SHA-256",
  "Source URL",
]];
const provenanceRows = payload.provenance.map((row) => {
  const url = row.input_file.includes("GSE14520")
    ? payload.source_urls.GSE14520
    : row.input_file.includes("GSE116174")
      ? payload.source_urls.GSE116174
      : "";
  return [row.input_file, row.role, row.sha256, url];
});
provenance.getRange(`A2:D${provenanceRows.length + 1}`).values = provenanceRows;
provenance.getRange(`A1:D${provenanceRows.length + 1}`).format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
provenance.getRange("A1:D1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
provenance.getRange("A1:A20").format.columnWidth = 68;
provenance.getRange("B1:B20").format.columnWidth = 46;
provenance.getRange("C1:C20").format.columnWidth = 68;
provenance.getRange("D1:D20").format.columnWidth = 62;
provenance.freezePanes.freezeRows(1);
provenance.tables.add(
  `A1:D${provenanceRows.length + 1}`,
  true,
  "Table3Provenance",
);

const definitionRows = [
  ["Term", "Definition"],
  ["Frozen gene-only component", "TCGA-derived 15-gene transport score with locked coefficients and no external refitting."],
  ["Outcome-blind scoring", "External survival outcomes were not used to generate the transported score."],
  ["Harrell C", "Concordance index based on comparable survival-time pairs."],
  ["Uno C", "IPCW concordance index evaluated at the cohort-specific truncation time τ."],
  ["HR per 1 SD", "Relative hazard per one within-cohort sample standard deviation higher transported score."],
  ["Frozen cutoff", "Median transported score in the locked TCGA derivation cohort, applied unchanged externally."],
  ["External recalibration", "Updating model scale, baseline hazard, coefficients, or threshold using external outcomes; none was performed."],
  ["Not M4 validation", "The external microarray data lack the full RNA-seq-plus-clinical feature space required to validate the combined RSF M4 model."],
];
definitions.getRange(`A1:B${definitionRows.length}`).values = definitionRows;
definitions.getRange(`A1:B${definitionRows.length}`).format = {
  font: { name: "Arial", size: 9, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
definitions.getRange("A1:B1").format = {
  fill: navy,
  font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
};
definitions.getRange("A1:A20").format.columnWidth = 32;
definitions.getRange("B1:B20").format.columnWidth = 100;
definitions.getRange("A2:B20").format.rowHeight = 30;
definitions.freezePanes.freezeRows(1);

const preview = await workbook.render({
  sheetName: "Table 3",
  range: "A1:I19",
  scale: 2,
  format: "png",
});
await fs.writeFile(PREVIEW, new Uint8Array(await preview.arrayBuffer()));

const inspect = await workbook.inspect({
  kind: "table",
  range: "Table 3!A1:I19",
  include: "values,formulas",
  tableMaxRows: 24,
  tableMaxCols: 11,
});
await fs.writeFile(
  path.join(ROOT, "TABLE3_INSPECT.ndjson"),
  inspect.ndjson,
  "utf8",
);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(ROOT, "TABLE3_FORMULA_ERROR_SCAN.ndjson"),
  formulaErrors.ndjson,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUTPUT);

console.log(JSON.stringify({
  status: "TABLE3_WORKBOOK_CREATED",
  output: OUTPUT,
  preview: PREVIEW,
  cohorts: payload.performance.map((row) => `${row.cohort}_${row.platform}`),
}, null, 2));
