import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.TABLE1_ROOT || SCRIPT_DIR;
const SOURCE_DIR = path.join(ROOT, "source_data");
const OUTPUT = path.join(ROOT, "Table_1_Cohort_Characteristics.xlsx");
const PREVIEW = path.join(ROOT, "Table_1_Preview.png");

const payload = JSON.parse(
  await fs.readFile(path.join(SOURCE_DIR, "table1_payload.json"), "utf8"),
);
const cohorts = payload.cohorts;
const cohortHeaders = [
  "TCGA-LIHC",
  "GSE14520 (GPL3921)",
  "GSE116174 (GPL570)",
];

const workbook = Workbook.create();
const table = workbook.worksheets.add("Table 1");
const numeric = workbook.worksheets.add("Numeric Source");
const provenance = workbook.worksheets.add("Provenance");
const definitions = workbook.worksheets.add("Definitions");
const excluded = workbook.worksheets.add("Excluded Cohort");

const navy = "#24364B";
const blue = "#536C8B";
const orange = "#C65D3A";
const paleBlue = "#E8EEF4";
const paleOrange = "#F6E8E2";
const paleGray = "#F4F5F7";
const midGray = "#66707C";
const rule = "#C7CDD4";
const text = "#20252B";

for (const sheet of [table, numeric, provenance, definitions, excluded]) {
  sheet.showGridLines = false;
}

table.mergeCells("A1:D1");
table.getRange("A1").values = [[
  "Table 1. Characteristics of the development and external transport cohorts",
]];
table.mergeCells("A2:D2");
table.getRange("A2").values = [[
  "Locked analysis populations used in Figures 2 and 3",
]];
table.getRange("A1:D1").format = {
  fill: navy,
  font: { name: "Arial", size: 14, bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
table.getRange("A2:D2").format = {
  fill: paleBlue,
  font: { name: "Arial", size: 9, italic: true, color: midGray },
  verticalAlignment: "center",
};
table.getRange("A1:D1").format.rowHeight = 28;
table.getRange("A2:D2").format.rowHeight = 22;

table.getRange("A4:D4").values = [[
  "Characteristic",
  ...cohortHeaders,
]];
table.getRange("A4:D4").format = {
  fill: blue,
  font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    bottom: { style: "medium", color: navy },
  },
};
table.getRange("A4").format.horizontalAlignment = "left";
table.getRange("A4:D4").format.rowHeight = 34;

const mainRows = [
  ["Analysis role", ...cohorts.map((x) => x.role)],
  ["Expression technology", ...cohorts.map((x) => x.expression_technology)],
  ["Patients, n", "", "", ""],
  ["Deaths, n (%)", "", "", ""],
  ["Observed OS/censoring time, median (IQR), months", "", "", ""],
  ["Age, median (IQR), years", "", "", ""],
  ["Male sex, n (%)", "", "", ""],
  ["AJCC/TNM stage, n (%)", "", "", ""],
  ["  I", "", "", ""],
  ["  II", "", "", ""],
  ["  III–IV", "", "", ""],
  ["  Missing", "", "", ""],
  ["Tumour grade, n (%)", "", "", ""],
  ["  G1–G2", "", "", ""],
  ["  G3–G4", "", "", ""],
  ["  Missing", "", "", ""],
  ["Locked metabolic genes available", "", "", ""],
];
table.getRange("A5:D21").values = mainRows;
table.getRange("A5:D21").format = {
  font: { name: "Arial", size: 9, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
table.getRange("B5:D21").format.horizontalAlignment = "center";
table.getRange("A5:A21").format.horizontalAlignment = "left";
table.getRange("B5:B21").format.fill = paleBlue;
table.getRange("C5:D21").format.fill = paleOrange;

for (const sectionRow of [12, 17]) {
  table.getRange(`A${sectionRow}:D${sectionRow}`).format = {
    fill: paleGray,
    font: { name: "Arial", size: 9, bold: true, color: navy },
    borders: {
      top: { style: "medium", color: navy },
      bottom: { style: "thin", color: rule },
    },
  };
}

table.getRange("A5:D6").format.rowHeight = 34;
table.getRange("A7:D21").format.rowHeight = 22;
table.getRange("A9:D9").format.rowHeight = 30;

const numericFields = [
  ["role", "Analysis role", "text"],
  ["expression_technology", "Expression technology", "text"],
  ["patients_n", "Patients", "n"],
  ["events_n", "Deaths", "n"],
  ["event_rate", "Death rate", "proportion"],
  ["observed_time_median_months", "Observed OS/censoring time median", "months"],
  ["observed_time_q1_months", "Observed OS/censoring time Q1", "months"],
  ["observed_time_q3_months", "Observed OS/censoring time Q3", "months"],
  ["age_median_years", "Age median", "years"],
  ["age_q1_years", "Age Q1", "years"],
  ["age_q3_years", "Age Q3", "years"],
  ["male_n", "Male sex", "n"],
  ["male_rate", "Male sex rate", "proportion"],
  ["stage_i_n", "Stage I", "n"],
  ["stage_i_rate", "Stage I rate", "proportion"],
  ["stage_ii_n", "Stage II", "n"],
  ["stage_ii_rate", "Stage II rate", "proportion"],
  ["stage_iii_iv_n", "Stage III–IV", "n"],
  ["stage_iii_iv_rate", "Stage III–IV rate", "proportion"],
  ["stage_missing_n", "Stage missing", "n"],
  ["stage_missing_rate", "Stage missing rate", "proportion"],
  ["grade_g1_g2_n", "Grade G1–G2", "n"],
  ["grade_g1_g2_rate", "Grade G1–G2 rate", "proportion"],
  ["grade_g3_g4_n", "Grade G3–G4", "n"],
  ["grade_g3_g4_rate", "Grade G3–G4 rate", "proportion"],
  ["grade_missing_n", "Grade missing", "n"],
  ["grade_missing_rate", "Grade missing rate", "proportion"],
  ["genes_available_n", "Locked genes available", "n"],
  ["genes_required_n", "Locked genes required", "n"],
];

numeric.getRange("A1:F1").values = [[
  "Field",
  "Display label",
  cohortHeaders[0],
  cohortHeaders[1],
  cohortHeaders[2],
  "Unit",
]];
const numericRows = numericFields.map(([field, label, unit]) => [
  field,
  label,
  ...cohorts.map((x) => x[field] ?? null),
  unit,
]);
numeric.getRange(`A2:F${numericRows.length + 1}`).values = numericRows;
numeric.getRange(`A1:F${numericRows.length + 1}`).format = {
  font: { name: "Arial", size: 8, color: text },
  borders: {
    insideHorizontal: { style: "thin", color: rule },
  },
};
numeric.getRange("A1:F1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
numeric.getRange("C2:E30").format.numberFormat = "0.000";
numeric.freezePanes.freezeRows(1);
numeric.tables.add(`A1:F${numericRows.length + 1}`, true, "Table1NumericSource");

const sourceRow = Object.fromEntries(
  numericFields.map(([field], index) => [field, index + 2]),
);
const sourceCol = ["C", "D", "E"];
const targetCol = ["B", "C", "D"];
const directFormula = (field, cohortIndex) =>
  `='Numeric Source'!${sourceCol[cohortIndex]}${sourceRow[field]}`;
const nPctFormula = (nField, rateField, cohortIndex) =>
  `=TEXT('Numeric Source'!${sourceCol[cohortIndex]}${sourceRow[nField]},"0")&" ("&TEXT('Numeric Source'!${sourceCol[cohortIndex]}${sourceRow[rateField]},"0.0%")&")"`;
const medianIqrFormula = (medianField, q1Field, q3Field, cohortIndex) =>
  `=TEXT('Numeric Source'!${sourceCol[cohortIndex]}${sourceRow[medianField]},"0.0")&" ("&TEXT('Numeric Source'!${sourceCol[cohortIndex]}${sourceRow[q1Field]},"0.0")&"–"&TEXT('Numeric Source'!${sourceCol[cohortIndex]}${sourceRow[q3Field]},"0.0")&")"`;

for (let i = 0; i < 3; i += 1) {
  table.getRange(`${targetCol[i]}7`).formulas = [[directFormula("patients_n", i)]];
  table.getRange(`${targetCol[i]}8`).formulas = [[nPctFormula("events_n", "event_rate", i)]];
  table.getRange(`${targetCol[i]}9`).formulas = [[
    medianIqrFormula(
      "observed_time_median_months",
      "observed_time_q1_months",
      "observed_time_q3_months",
      i,
    ),
  ]];
  table.getRange(`${targetCol[i]}10`).formulas = [[
    medianIqrFormula("age_median_years", "age_q1_years", "age_q3_years", i),
  ]];
  table.getRange(`${targetCol[i]}11`).formulas = [[nPctFormula("male_n", "male_rate", i)]];
  table.getRange(`${targetCol[i]}13`).formulas = [[nPctFormula("stage_i_n", "stage_i_rate", i)]];
  table.getRange(`${targetCol[i]}14`).formulas = [[nPctFormula("stage_ii_n", "stage_ii_rate", i)]];
  table.getRange(`${targetCol[i]}15`).formulas = [[nPctFormula("stage_iii_iv_n", "stage_iii_iv_rate", i)]];
  table.getRange(`${targetCol[i]}16`).formulas = [[nPctFormula("stage_missing_n", "stage_missing_rate", i)]];
  if (i === 0) {
    table.getRange(`${targetCol[i]}18`).formulas = [[nPctFormula("grade_g1_g2_n", "grade_g1_g2_rate", i)]];
    table.getRange(`${targetCol[i]}19`).formulas = [[nPctFormula("grade_g3_g4_n", "grade_g3_g4_rate", i)]];
    table.getRange(`${targetCol[i]}20`).formulas = [[nPctFormula("grade_missing_n", "grade_missing_rate", i)]];
  } else {
    table.getRange(`${targetCol[i]}18:${targetCol[i]}20`).values = [["NA"], ["NA"], ["NA"]];
  }
  table.getRange(`${targetCol[i]}21`).formulas = [[
    `=TEXT('Numeric Source'!${sourceCol[i]}${sourceRow.genes_available_n},"0")&"/"&TEXT('Numeric Source'!${sourceCol[i]}${sourceRow.genes_required_n},"0")`,
  ]];
}

const notes = [
  "Values are n (%) unless otherwise indicated. IQR, interquartile range; OS, overall survival.",
  "Observed time is the recorded OS/censoring duration, not a reverse Kaplan–Meier estimate of follow-up.",
  "The two microarray cohorts are secondary exploratory gene-only transport analyses and are not formal external validation of the TCGA combined RSF model.",
  "GSE14520-GPL571 (N=21) was excluded before analysis because the sample size was insufficient.",
  "TCGA sex was matched by submitter ID to the cBioPortal hcc_tcga_gdc clinical table; 363/363 cases matched.",
];
for (let i = 0; i < notes.length; i += 1) {
  const row = 23 + i;
  table.mergeCells(`A${row}:D${row}`);
  table.getRange(`A${row}`).values = [[notes[i]]];
}
table.getRange("A23:D27").format = {
  font: { name: "Arial", size: 8, color: midGray },
  wrapText: true,
  verticalAlignment: "center",
};
table.getRange("A23:D27").format.rowHeight = 24;
table.getRange("A23:D23").format.borders = {
  top: { style: "medium", color: navy },
};

table.getRange("A1:A27").format.columnWidth = 43;
table.getRange("B1:B27").format.columnWidth = 28;
table.getRange("C1:C27").format.columnWidth = 28;
table.getRange("D1:D27").format.columnWidth = 31;
table.freezePanes.freezeRows(4);

const provenanceHeaders = ["Input file", "Source URL or role", "SHA-256"];
provenance.getRange("A1:C1").values = [provenanceHeaders];
const provenanceRows = payload.provenance.map((row) => [
  row.input_file,
  row.source_url_or_role,
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
provenance.getRange("A1:A20").format.columnWidth = 58;
provenance.getRange("B1:B20").format.columnWidth = 70;
provenance.getRange("C1:C20").format.columnWidth = 68;
provenance.freezePanes.freezeRows(1);
provenance.tables.add(`A1:C${provenanceRows.length + 1}`, true, "Table1Provenance");

const definitionRows = [
  ["Term", "Definition"],
  ["Analysis population", "Patients appearing in the locked modelling or external-evaluation files."],
  ["Deaths", "Overall-survival event indicator equal to 1."],
  ["Observed OS/censoring time", "Recorded duration from diagnosis or study baseline to death or censoring."],
  ["Stage I/II/III–IV", "AJCC or TNM stage collapsed by leading Roman numeral; substages retain their parent stage."],
  ["Tumour grade", "TCGA G1–G2 and G3–G4; grade was not available in the locked external analysis inputs."],
  ["NA", "Not available in the locked analysis input; no value was imputed."],
  ["Secondary exploratory transport", "Frozen gene-only cross-platform evaluation; not external validation of the combined RSF model."],
  ["GPL571", "Not analysed because only 21 complete-OS cases were available."],
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
definitions.getRange("B1:B20").format.columnWidth = 95;
definitions.getRange("A2:B20").format.rowHeight = 28;
definitions.freezePanes.freezeRows(1);

excluded.getRange("A1:F1").values = [[
  "Cohort",
  "Platform",
  "Complete OS cases",
  "Events",
  "Analysis status",
  "Reason",
]];
const excludedRows = payload.excluded_cohorts.map((row) => [
  row.cohort,
  row.platform,
  row.complete_os_cases,
  row.events,
  row.analysis_status,
  row.reason,
]);
excluded.getRange(`A2:F${excludedRows.length + 1}`).values = excludedRows;
excluded.getRange(`A1:F${excludedRows.length + 1}`).format = {
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
excluded.getRange("A1:B5").format.columnWidth = 18;
excluded.getRange("C1:E5").format.columnWidth = 22;
excluded.getRange("F1:F5").format.columnWidth = 55;

const preview = await workbook.render({
  sheetName: "Table 1",
  range: "A1:D27",
  scale: 2,
  format: "png",
});
await fs.writeFile(PREVIEW, new Uint8Array(await preview.arrayBuffer()));

const inspect = await workbook.inspect({
  kind: "table",
  range: "Table 1!A1:D27",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 6,
});
await fs.writeFile(
  path.join(ROOT, "TABLE1_INSPECT.ndjson"),
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
  path.join(ROOT, "TABLE1_FORMULA_ERROR_SCAN.ndjson"),
  formulaErrors.ndjson,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUTPUT);

console.log(JSON.stringify({
  status: "TABLE1_WORKBOOK_CREATED",
  output: OUTPUT,
  preview: PREVIEW,
  cohorts: cohortHeaders,
}, null, 2));
