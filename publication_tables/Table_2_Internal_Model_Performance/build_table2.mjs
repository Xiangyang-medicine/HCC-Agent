import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.TABLE2_ROOT || SCRIPT_DIR;
const SOURCE_DIR = path.join(ROOT, "source_data");
const OUTPUT = path.join(ROOT, "Table_2_Internal_Model_Performance.xlsx");
const PREVIEW = path.join(ROOT, "Table_2_Preview.png");

const payload = JSON.parse(
  await fs.readFile(path.join(SOURCE_DIR, "table2_payload.json"), "utf8"),
);

const workbook = Workbook.create();
const table = workbook.worksheets.add("Table 2");
const numeric = workbook.worksheets.add("Numeric Performance");
const comparisons = workbook.worksheets.add("Paired Comparisons");
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
const green = "#DDEBDD";
const red = "#F5DEDE";

for (const sheet of [table, numeric, comparisons, provenance, definitions]) {
  sheet.showGridLines = false;
}

table.mergeCells("A1:J1");
table.getRange("A1").values = [[payload.table_title]];
table.mergeCells("A2:J2");
table.getRange("A2").values = [[payload.subtitle]];
table.getRange("A1:J1").format = {
  fill: navy,
  font: { name: "Arial", size: 14, bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
table.getRange("A2:J2").format = {
  fill: paleBlue,
  font: { name: "Arial", size: 9, italic: true, color: midGray },
  verticalAlignment: "center",
};
table.getRange("A1:J1").format.rowHeight = 28;
table.getRange("A2:J2").format.rowHeight = 22;

table.mergeCells("A4:J4");
table.getRange("A4").values = [["A. Model performance across outer test folds"]];
table.getRange("A4:J4").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};

table.getRange("A5:J5").values = [[
  "Model",
  "Predictor set",
  "Algorithm",
  "Harrell C",
  "Uno C",
  "AUC\n12 months",
  "AUC\n36 months",
  "AUC\n60 months",
  "IBS",
  "Model role",
]];
table.getRange("A5:J5").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    bottom: { style: "medium", color: navy },
  },
};
table.getRange("A5:C5").format.horizontalAlignment = "left";
table.getRange("J5").format.horizontalAlignment = "left";
table.getRange("A5:J5").format.rowHeight = 35;

const performanceTemplate = payload.performance.map((row) => [
  `${row.model_id} ${row.display_name}`,
  row.predictors,
  row.algorithm,
  "",
  "",
  "",
  "",
  "",
  "",
  row.role,
]);
table.getRange("A6:J10").values = performanceTemplate;
table.getRange("A6:J10").format = {
  font: { name: "Arial", size: 8, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
table.getRange("D6:I10").format.horizontalAlignment = "center";
table.getRange("A6:C10").format.horizontalAlignment = "left";
table.getRange("J6:J10").format.horizontalAlignment = "left";
table.getRange("A6:J10").format.rowHeight = 34;
table.getRange("A6:J6").format.fill = paleBlue;
table.getRange("A9:J9").format.fill = paleOrange;
table.getRange("A9:J9").format.font = {
  name: "Arial",
  size: 8,
  bold: true,
  color: text,
};

const numericHeaders = [
  "Model key",
  "Model ID",
  "Display name",
  "Predictors",
  "Algorithm",
  "Role",
  "Outer folds",
  "Harrell C mean",
  "Harrell C SD",
  "Uno C mean",
  "Uno C SD",
  "AUC 12m mean",
  "AUC 12m SD",
  "AUC 36m mean",
  "AUC 36m SD",
  "AUC 60m mean",
  "AUC 60m SD",
  "IBS mean",
  "IBS SD",
  "Brier 12m mean",
  "Brier 36m mean",
  "Brier 60m mean",
];
numeric.getRange("A1:V1").values = [numericHeaders];
const numericRows = payload.performance.map((row) => [
  row.model_key,
  row.model_id,
  row.display_name,
  row.predictors,
  row.algorithm,
  row.role,
  row.n_outer_folds,
  row.harrell_c_mean,
  row.harrell_c_sd,
  row.uno_c_mean,
  row.uno_c_sd,
  row.auc_12m_mean,
  row.auc_12m_sd,
  row.auc_36m_mean,
  row.auc_36m_sd,
  row.auc_60m_mean,
  row.auc_60m_sd,
  row.ibs_mean,
  row.ibs_sd,
  row.brier_12m_mean,
  row.brier_36m_mean,
  row.brier_60m_mean,
]);
numeric.getRange("A2:V6").values = numericRows;
numeric.getRange("A1:V6").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
numeric.getRange("A1:V1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
numeric.getRange("G2:V6").format.numberFormat = "0.000000";
numeric.getRange("A1:F10").format.columnWidth = 29;
numeric.getRange("G1:V10").format.columnWidth = 15;
numeric.freezePanes.freezeRows(1);
numeric.tables.add("A1:V6", true, "Table2NumericPerformance");

const meanSdFormula = (meanCol, sdCol, row) =>
  `=TEXT('Numeric Performance'!${meanCol}${row},"0.000")&" ("&TEXT('Numeric Performance'!${sdCol}${row},"0.000")&")"`;
for (let index = 0; index < 5; index += 1) {
  const targetRow = index + 6;
  const sourceRow = index + 2;
  table.getRange(`D${targetRow}`).formulas = [[meanSdFormula("H", "I", sourceRow)]];
  table.getRange(`E${targetRow}`).formulas = [[meanSdFormula("J", "K", sourceRow)]];
  table.getRange(`F${targetRow}`).formulas = [[meanSdFormula("L", "M", sourceRow)]];
  table.getRange(`G${targetRow}`).formulas = [[meanSdFormula("N", "O", sourceRow)]];
  table.getRange(`H${targetRow}`).formulas = [[meanSdFormula("P", "Q", sourceRow)]];
  table.getRange(`I${targetRow}`).formulas = [[meanSdFormula("R", "S", sourceRow)]];
}

table.mergeCells("A12:J12");
table.getRange("A12").values = [[
  "B. Prespecified patient-level paired bootstrap comparisons",
]];
table.getRange("A12:J12").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};

table.getRange("A13:J13").values = [[
  "Comparison",
  "Metric",
  "Mean difference",
  "95% CI",
  "Raw p",
  "Adjusted p",
  "Adjusted result",
  "Valid resamples",
  "Patients",
  "IPCW source",
]];
table.getRange("A13:J13").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    bottom: { style: "medium", color: navy },
  },
};
table.getRange("A13:B13").format.horizontalAlignment = "left";
table.getRange("J13").format.horizontalAlignment = "left";
table.getRange("A13:J13").format.rowHeight = 35;

const comparisonRows = payload.comparisons.map((row) => [
  row.comparison,
  row.metric,
  "",
  "",
  "",
  "",
  row.significant_adjusted ? "Significant" : "Not significant",
  row.iterations_valid,
  row.n_patients,
  row.ipcw_source === "outer_training_fold" ? "Outer training fold" : "Not applicable",
]);
table.getRange("A14:J21").values = comparisonRows;
table.getRange("A14:J21").format = {
  font: { name: "Arial", size: 8, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
table.getRange("C14:I21").format.horizontalAlignment = "center";
table.getRange("A14:B21").format.horizontalAlignment = "left";
table.getRange("J14:J21").format.horizontalAlignment = "left";
table.getRange("A14:J21").format.rowHeight = 28;

const comparisonHeaders = [
  "Comparison",
  "Metric",
  "Metric key",
  "Model A",
  "Model B",
  "Mean difference",
  "CI lower",
  "CI upper",
  "Raw p",
  "Adjusted p",
  "Significant adjusted",
  "Valid resamples",
  "Patients",
  "Repeats",
  "Folds",
  "IPCW source",
];
comparisons.getRange("A1:P1").values = [comparisonHeaders];
const comparisonNumericRows = payload.comparisons.map((row) => [
  row.comparison,
  row.metric,
  row.metric_key,
  row.model_a,
  row.model_b,
  row.mean_difference,
  row.ci_lower,
  row.ci_upper,
  row.p_value_raw,
  row.p_value_adjusted,
  row.significant_adjusted,
  row.iterations_valid,
  row.n_patients,
  row.n_repeats,
  row.n_folds,
  row.ipcw_source,
]);
comparisons.getRange("A2:P9").values = comparisonNumericRows;
comparisons.getRange("A1:P9").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
comparisons.getRange("A1:P1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
comparisons.getRange("F2:J9").format.numberFormat = "0.000000";
comparisons.getRange("A1:E12").format.columnWidth = 25;
comparisons.getRange("F1:P12").format.columnWidth = 16;
comparisons.freezePanes.freezeRows(1);
comparisons.tables.add("A1:P9", true, "Table2PairedComparisons");

const signedFormula = (sourceCol, sourceRow) =>
  `=IF('Paired Comparisons'!${sourceCol}${sourceRow}>=0,"+"&TEXT('Paired Comparisons'!${sourceCol}${sourceRow},"0.000"),TEXT('Paired Comparisons'!${sourceCol}${sourceRow},"0.000"))`;
const ciFormula = (sourceRow) =>
  `="["&IF('Paired Comparisons'!G${sourceRow}>=0,"+"&TEXT('Paired Comparisons'!G${sourceRow},"0.000"),TEXT('Paired Comparisons'!G${sourceRow},"0.000"))&" to "&IF('Paired Comparisons'!H${sourceRow}>=0,"+"&TEXT('Paired Comparisons'!H${sourceRow},"0.000"),TEXT('Paired Comparisons'!H${sourceRow},"0.000"))&"]"`;
const pFormula = (sourceCol, sourceRow) =>
  `=IF('Paired Comparisons'!${sourceCol}${sourceRow}<0.001,"<0.001",TEXT('Paired Comparisons'!${sourceCol}${sourceRow},"0.000"))`;

for (let index = 0; index < 8; index += 1) {
  const targetRow = index + 14;
  const sourceRow = index + 2;
  table.getRange(`C${targetRow}`).formulas = [[signedFormula("F", sourceRow)]];
  table.getRange(`D${targetRow}`).formulas = [[ciFormula(sourceRow)]];
  table.getRange(`E${targetRow}`).formulas = [[pFormula("I", sourceRow)]];
  table.getRange(`F${targetRow}`).formulas = [[pFormula("J", sourceRow)]];
  const significant = payload.comparisons[index].significant_adjusted;
  table.getRange(`G${targetRow}`).format.fill = significant ? green : paleGray;
  if (significant) {
    table.getRange(`A${targetRow}:J${targetRow}`).format.font = {
      name: "Arial",
      size: 8,
      bold: true,
      color: text,
    };
  }
}

const notes = [
  "Section A values are mean (SD) across 25 outer test folds; higher Harrell C, Uno C, and AUC are better, whereas lower IBS is better.",
  "Section B uses a patient-level paired bootstrap with 1,000 valid resamples; 95% CIs are percentile intervals.",
  "Two-sided p values were Bonferroni-adjusted within each metric family of four formal comparisons.",
  "Uno C used censoring weights from the corresponding outer training fold; IPCW, inverse probability of censoring weighting.",
  "M4 was descriptively strongest and remains the provisional primary candidate, but was not significantly better than M1 after adjustment.",
  "M5 was significantly worse than M1 for Uno C after adjustment; no other formal comparison was significant after adjustment.",
];
for (let index = 0; index < notes.length; index += 1) {
  const row = index + 23;
  table.mergeCells(`A${row}:J${row}`);
  table.getRange(`A${row}`).values = [[notes[index]]];
}
table.getRange("A23:J28").format = {
  font: { name: "Arial", size: 8, color: midGray },
  wrapText: true,
  verticalAlignment: "center",
};
table.getRange("A23:J28").format.rowHeight = 23;
table.getRange("A23:J23").format.borders = {
  top: { style: "medium", color: navy },
};

table.getRange("A1:A30").format.columnWidth = 25;
table.getRange("B1:B30").format.columnWidth = 31;
table.getRange("C1:C30").format.columnWidth = 24;
table.getRange("D1:I30").format.columnWidth = 16;
table.getRange("J1:J30").format.columnWidth = 31;
table.freezePanes.freezeRows(5);

provenance.getRange("A1:C1").values = [["Input file", "Role", "SHA-256"]];
const provenanceRows = payload.provenance.map((row) => [
  row.input_file,
  row.role,
  row.sha256,
]);
provenance.getRange(`A2:C${provenanceRows.length + 1}`).values = provenanceRows;
provenance.getRange(`A1:C${provenanceRows.length + 1}`).format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
provenance.getRange("A1:C1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
provenance.getRange("A1:A20").format.columnWidth = 68;
provenance.getRange("B1:B20").format.columnWidth = 48;
provenance.getRange("C1:C20").format.columnWidth = 68;
provenance.freezePanes.freezeRows(1);
provenance.tables.add(
  `A1:C${provenanceRows.length + 1}`,
  true,
  "Table2Provenance",
);

const definitionRows = [
  ["Term", "Definition"],
  ["Outer test fold", "Held-out fold used only for performance evaluation in repeated nested cross-validation."],
  ["Harrell C", "Concordance index based on comparable survival-time pairs."],
  ["Uno C", "IPCW concordance index using censoring weights estimated from outer training data."],
  ["Time-dependent AUC", "Cumulative/dynamic discrimination at the stated month horizon."],
  ["IBS", "Integrated Brier score; lower values indicate better overall prediction error."],
  ["Paired bootstrap", "The same resampled patients are used for both models and all repeats before averaging repeat-specific differences."],
  ["Adjusted p", "Two-sided p value after Bonferroni correction across four formal comparisons within a metric family."],
  ["Provisional primary candidate", "Selected for subsequent evaluation on descriptive performance and stability, without a claim of statistical superiority over M1."],
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
definitions.getRange("A1:A20").format.columnWidth = 31;
definitions.getRange("B1:B20").format.columnWidth = 100;
definitions.getRange("A2:B20").format.rowHeight = 30;
definitions.freezePanes.freezeRows(1);

const preview = await workbook.render({
  sheetName: "Table 2",
  range: "A1:J28",
  scale: 2,
  format: "png",
});
await fs.writeFile(PREVIEW, new Uint8Array(await preview.arrayBuffer()));

const inspect = await workbook.inspect({
  kind: "table",
  range: "Table 2!A1:J28",
  include: "values,formulas",
  tableMaxRows: 32,
  tableMaxCols: 12,
});
await fs.writeFile(
  path.join(ROOT, "TABLE2_INSPECT.ndjson"),
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
  path.join(ROOT, "TABLE2_FORMULA_ERROR_SCAN.ndjson"),
  formulaErrors.ndjson,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUTPUT);

console.log(JSON.stringify({
  status: "TABLE2_WORKBOOK_CREATED",
  output: OUTPUT,
  preview: PREVIEW,
  models: payload.performance.map((row) => row.model_id),
  formalComparisons: payload.comparisons.length,
}, null, 2));
