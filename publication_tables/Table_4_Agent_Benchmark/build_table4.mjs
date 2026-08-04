import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.TABLE4_ROOT || SCRIPT_DIR;
const SOURCE_DIR = path.join(ROOT, "source_data");
const OUTPUT = path.join(ROOT, "Table_4_Agent_Benchmark.xlsx");
const PREVIEW = path.join(ROOT, "Table_4_Preview.png");

const payload = JSON.parse(
  await fs.readFile(path.join(SOURCE_DIR, "table4_payload.json"), "utf8"),
);

const workbook = Workbook.create();
const table = workbook.worksheets.add("Table 4");
const systems = workbook.worksheets.add("System Definitions");
const primary = workbook.worksheets.add("Primary Endpoint");
const trace = workbook.worksheets.add("Traceability");
const planning = workbook.worksheets.add("Planning Audit");
const ablations = workbook.worksheets.add("Ablations");
const faults = workbook.worksheets.add("Fault Handling");
const corrections = workbook.worksheets.add("Metric Audit");
const provenance = workbook.worksheets.add("Provenance");
const definitions = workbook.worksheets.add("Definitions");

const navy = "#24364B";
const blue = "#536C8B";
const orange = "#C65D3A";
const paleBlue = "#E8EEF4";
const paleOrange = "#F6E8E2";
const paleGray = "#F4F5F7";
const paleGreen = "#DDEBDD";
const midGray = "#66707C";
const rule = "#C7CDD4";
const text = "#20252B";

for (const sheet of [
  table,
  systems,
  primary,
  trace,
  planning,
  ablations,
  faults,
  corrections,
  provenance,
  definitions,
]) {
  sheet.showGridLines = false;
}

table.mergeCells("A1:H1");
table.getRange("A1").values = [[payload.table_title]];
table.mergeCells("A2:H2");
table.getRange("A2").values = [[payload.subtitle]];
table.getRange("A1:H1").format = {
  fill: navy,
  font: { name: "Arial", size: 14, bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
table.getRange("A2:H2").format = {
  fill: paleBlue,
  font: { name: "Arial", size: 9, italic: true, color: midGray },
  verticalAlignment: "center",
};
table.getRange("A1:H1").format.rowHeight = 28;
table.getRange("A2:H2").format.rowHeight = 22;

table.mergeCells("A4:H4");
table.getRange("A4").values = [["A. Systems compared in the primary analysis"]];
table.getRange("A4:H4").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};
table.getRange("A5:H5").values = [[
  "System",
  "Controller",
  "Frozen tools",
  "Internal verifier",
  "Conditional repair",
  "Terminal states",
  "Cases",
  "Runs",
]];
table.getRange("A5:H5").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: navy } },
};
table.getRange("A5:F5").format.horizontalAlignment = "left";
table.getRange("A5:H5").format.rowHeight = 34;
table.getRange("A6:H7").values = payload.systems.map((row) => [
  row.display_name,
  row.controller,
  row.tools,
  row.internal_verifier,
  row.conditional_repair,
  row.terminal_states,
  100,
  300,
]);
table.getRange("A6:H7").format = {
  font: { name: "Arial", size: 8, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
table.getRange("G6:H7").format.horizontalAlignment = "center";
table.getRange("A6:F7").format.horizontalAlignment = "left";
table.getRange("A6:H6").format.fill = paleBlue;
table.getRange("A7:H7").format.fill = paleOrange;
table.getRange("A7:H7").format.font = {
  name: "Arial",
  size: 8,
  bold: true,
  color: text,
};
table.getRange("A6:H7").format.rowHeight = 44;

systems.getRange("A1:G1").values = [[
  "System",
  "Display name",
  "Controller",
  "Tools",
  "Internal verifier",
  "Conditional repair",
  "Terminal states",
]];
systems.getRange("A2:G3").values = payload.systems.map((row) => [
  row.system,
  row.display_name,
  row.controller,
  row.tools,
  row.internal_verifier,
  row.conditional_repair,
  row.terminal_states,
]);
systems.getRange("A1:G3").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
systems.getRange("A1:G1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
systems.getRange("A1:G5").format.columnWidth = 32;
systems.freezePanes.freezeRows(1);
systems.tables.add("A1:G3", true, "Table4SystemDefinitions");

table.mergeCells("A9:H9");
table.getRange("A9").values = [[
  "B. Prespecified primary endpoint and comparable technical metrics",
]];
table.getRange("A9:H9").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};
table.getRange("A10:H10").values = [[
  "Endpoint",
  "Reference standard",
  "B2 estimate\n(95% CI)",
  "B4 estimate\n(95% CI)",
  "B4−B2 absolute difference, pp\n(95% CI)",
  "p value",
  "Analysis status",
  "Analysis unit",
]];
table.getRange("A10:H10").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: navy } },
};
table.getRange("A10:B10").format.horizontalAlignment = "left";
table.getRange("G10:H10").format.horizontalAlignment = "left";
table.getRange("A10:H10").format.rowHeight = 38;

const traceMetrics = [
  "exact_extractive_claim_support",
  "assigned_passage_citation_id_validity",
  "exact_three_run_agreement",
];
const primaryB2 = payload.primary.find((row) => row.system.startsWith("B2_"));
const primaryB4 = payload.primary.find((row) => row.system.startsWith("B4_"));
const metricRows = [
  {
    label: "Frozen independently implemented composite pass",
    reference: "Prespecified deterministic composite scorer",
    b2: primaryB2,
    b4: primaryB4,
    difference: payload.paired,
    pValue: payload.paired.p_value,
    status: "Prespecified confirmatory",
    unit: "300 paired runs",
  },
  ...traceMetrics.map((metric) => {
    const b2 = payload.traceability.find(
      (row) => row.metric === metric && row.system.startsWith("B2_"),
    );
    const b4 = payload.traceability.find(
      (row) => row.metric === metric && row.system.startsWith("B4_"),
    );
    return {
      label: b2.metric_label,
      reference: b2.reference_standard,
      b2,
      b4,
      difference: null,
      pValue: null,
      status: "Descriptive technical endpoint",
      unit: metric === "exact_three_run_agreement" ? "100 cases" : "300 runs",
    };
  }),
];
table.getRange("A11:H14").values = metricRows.map((row) => [
  row.label,
  row.reference,
  "",
  "",
  "",
  "",
  row.status,
  row.unit,
]);
table.getRange("A11:H14").format = {
  font: { name: "Arial", size: 8, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
table.getRange("C11:F14").format.horizontalAlignment = "center";
table.getRange("A11:B14").format.horizontalAlignment = "left";
table.getRange("G11:H14").format.horizontalAlignment = "left";
table.getRange("A11:H11").format.fill = paleOrange;
table.getRange("A11:H11").format.font = {
  name: "Arial",
  size: 8,
  bold: true,
  color: text,
};
table.getRange("A11:H14").format.rowHeight = 38;

primary.getRange("A1:K1").values = [[
  "System",
  "Display name",
  "Endpoint",
  "Status",
  "Cases",
  "Runs",
  "Successes",
  "Rate",
  "CI lower",
  "CI upper",
  "Rate check",
]];
primary.getRange("A2:K3").values = payload.primary.map((row) => [
  row.system,
  row.display_name,
  row.endpoint,
  row.analysis_status,
  row.n_cases,
  row.n_runs,
  row.successes,
  row.rate,
  row.ci_lower,
  row.ci_upper,
  null,
]);
primary.getRange("K2:K3").formulas = [
  ["=G2/F2"],
  ["=G3/F3"],
];
primary.getRange("A5:N5").values = [[
  "Comparison",
  "Effect definition",
  "Difference",
  "CI lower",
  "CI upper",
  "p value",
  "Cases",
  "Bootstrap",
  "Permutations",
  "Both pass",
  "B4 only",
  "B2 only",
  "Both fail",
  "Paired runs",
]];
primary.getRange("A6:N6").values = [[
  payload.paired.comparison,
  payload.paired.effect_definition,
  payload.paired.difference,
  payload.paired.ci_lower,
  payload.paired.ci_upper,
  payload.paired.p_value,
  payload.paired.n_cases,
  payload.paired.n_bootstrap,
  payload.paired.n_permutation,
  payload.paired.both_pass,
  payload.paired.b4_only_pass,
  payload.paired.b2_only_pass,
  payload.paired.both_fail,
  payload.paired.n_paired_runs,
]];
primary.getRange("A1:N6").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
primary.getRange("A1:K1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
primary.getRange("A5:N5").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
primary.getRange("H2:K3").format.numberFormat = "0.000000";
primary.getRange("C6:F6").format.numberFormat = "0.000000";
primary.getRange("A1:N8").format.columnWidth = 18;
primary.freezePanes.freezeRows(1);

trace.getRange("A1:J1").values = [[
  "System",
  "Display name",
  "Metric",
  "Metric label",
  "Reference standard",
  "Cases",
  "Runs",
  "Value",
  "CI lower",
  "CI upper",
]];
trace.getRange("A2:J7").values = payload.traceability.map((row) => [
  row.system,
  row.display_name,
  row.metric,
  row.metric_label,
  row.reference_standard,
  row.n_cases,
  row.n_runs,
  row.value,
  row.ci_lower,
  row.ci_upper,
]);
trace.getRange("A1:J7").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
trace.getRange("A1:J1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
trace.getRange("H2:J7").format.numberFormat = "0.000000";
trace.getRange("A1:J9").format.columnWidth = 22;
trace.freezePanes.freezeRows(1);
trace.tables.add("A1:J7", true, "Table4Traceability");

const estimateCiFormula = (sheetName, valueCol, lowCol, highCol, sourceRow) =>
  `=TEXT('${sheetName}'!${valueCol}${sourceRow},"0.0%")&" ("&TEXT('${sheetName}'!${lowCol}${sourceRow},"0.0%")&"–"&TEXT('${sheetName}'!${highCol}${sourceRow},"0.0%")&")"`;
table.getRange("C11").formulas = [[
  estimateCiFormula("Primary Endpoint", "H", "I", "J", 2),
]];
table.getRange("D11").formulas = [[
  estimateCiFormula("Primary Endpoint", "H", "I", "J", 3),
]];
table.getRange("E11").formulas = [[
  `=TEXT('Primary Endpoint'!C6*100,"+0.0;-0.0;0.0")&" pp ("&TEXT('Primary Endpoint'!D6*100,"+0.0;-0.0;0.0")&" to "&TEXT('Primary Endpoint'!E6*100,"+0.0;-0.0;0.0")&")"`,
]];
table.getRange("F11").formulas = [[
  `=IF('Primary Endpoint'!F6<0.001,"<0.001",TEXT('Primary Endpoint'!F6,"0.000"))`,
]];

for (let index = 0; index < 3; index += 1) {
  const targetRow = index + 12;
  const metric = traceMetrics[index];
  const b2Row = payload.traceability.findIndex(
    (row) => row.metric === metric && row.system.startsWith("B2_"),
  ) + 2;
  const b4Row = payload.traceability.findIndex(
    (row) => row.metric === metric && row.system.startsWith("B4_"),
  ) + 2;
  table.getRange(`C${targetRow}`).formulas = [[
    estimateCiFormula("Traceability", "H", "I", "J", b2Row),
  ]];
  table.getRange(`D${targetRow}`).formulas = [[
    estimateCiFormula("Traceability", "H", "I", "J", b4Row),
  ]];
  table.getRange(`E${targetRow}:F${targetRow}`).values = [["Not tested", "—"]];
}

table.mergeCells("A16:H16");
table.getRange("A16").values = [[
  "C. Post-hoc action-specification audit",
]];
table.getRange("A16:H16").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};
table.getRange("A17:H17").values = [[
  "System",
  "Runs",
  "Initially invalid",
  "Initial invalid rate",
  "Repaired",
  "Repair rate among initially invalid",
  "Finally invalid",
  "Final invalid rate",
]];
table.getRange("A17:H17").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: navy } },
};
table.getRange("A17").format.horizontalAlignment = "left";
table.getRange("A17:H17").format.rowHeight = 38;
table.getRange("A18:H19").values = payload.planning.map((row) => [
  row.display_name,
  row.n_runs,
  row.initial_invalid_n,
  row.initial_invalid_rate,
  row.repaired_n,
  row.repair_rate_among_initial_invalid,
  row.final_invalid_n,
  row.final_invalid_rate,
]);
table.getRange("A18:H19").format = {
  font: { name: "Arial", size: 8, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
table.getRange("B18:H19").format.horizontalAlignment = "center";
table.getRange("D18:D19").format.numberFormat = "0.0%";
table.getRange("F18:F19").format.numberFormat = "0.0%";
table.getRange("H18:H19").format.numberFormat = "0.0%";
table.getRange("A18:H18").format.fill = paleBlue;
table.getRange("A19:H19").format.fill = paleOrange;
table.getRange("A18:H19").format.rowHeight = 30;

planning.getRange("A1:I1").values = [[
  "System",
  "Display name",
  "Runs",
  "Initially invalid n",
  "Initial invalid rate",
  "Repaired n",
  "Repair rate among invalid",
  "Finally invalid n",
  "Final invalid rate",
]];
planning.getRange("A2:I3").values = payload.planning.map((row) => [
  row.system,
  row.display_name,
  row.n_runs,
  row.initial_invalid_n,
  row.initial_invalid_rate,
  row.repaired_n,
  row.repair_rate_among_initial_invalid,
  row.final_invalid_n,
  row.final_invalid_rate,
]);
planning.getRange("A1:I3").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
planning.getRange("A1:I1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
planning.getRange("E2:E3").format.numberFormat = "0.0%";
planning.getRange("G2:G3").format.numberFormat = "0.0%";
planning.getRange("I2:I3").format.numberFormat = "0.0%";
planning.getRange("A1:I5").format.columnWidth = 23;
planning.freezePanes.freezeRows(1);
planning.tables.add("A1:I3", true, "Table4PlanningAudit");

table.mergeCells("A21:H21");
table.getRange("A21").values = [[
  "D. Component ablations relative to full B4",
]];
table.getRange("A21:H21").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};
table.getRange("A22:H22").values = [[
  "Removed component",
  "Estimated composite pass",
  "Change vs full B4, pp",
  "95% CI, pp",
  "Raw p",
  "Holm-adjusted p",
  "Cases",
  "Inference",
]];
table.getRange("A22:H22").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: navy } },
};
table.getRange("A22").format.horizontalAlignment = "left";
table.getRange("H22").format.horizontalAlignment = "left";
table.getRange("A22:H22").format.rowHeight = 36;
table.getRange("A23:H26").values = payload.ablations.map((row) => [
  row.display_name,
  row.estimated_pass_rate,
  "",
  "",
  "",
  "",
  row.n_cases,
  row.p_holm < 0.05 ? "Significant reduction" : "No significant reduction",
]);
table.getRange("A23:H26").format = {
  font: { name: "Arial", size: 8, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
table.getRange("B23:G26").format.horizontalAlignment = "center";
table.getRange("B23:B26").format.numberFormat = "0.0%";
table.getRange("A23:H26").format.rowHeight = 30;
for (let index = 0; index < 4; index += 1) {
  const targetRow = index + 23;
  const sourceRow = index + 2;
  table.getRange(`C${targetRow}`).formulas = [[
    `=TEXT('Ablations'!E${sourceRow}*100,"+0.0;-0.0;0.0")&" pp"`,
  ]];
  table.getRange(`D${targetRow}`).formulas = [[
    `=TEXT('Ablations'!F${sourceRow}*100,"+0.0;-0.0;0.0")&" to "&TEXT('Ablations'!G${sourceRow}*100,"+0.0;-0.0;0.0")`,
  ]];
  table.getRange(`E${targetRow}`).formulas = [[
    `=IF('Ablations'!H${sourceRow}<0.001,"<0.001",TEXT('Ablations'!H${sourceRow},"0.000"))`,
  ]];
  table.getRange(`F${targetRow}`).formulas = [[
    `=IF('Ablations'!I${sourceRow}<0.001,"<0.001",TEXT('Ablations'!I${sourceRow},"0.000"))`,
  ]];
  if (payload.ablations[index].p_holm < 0.05) {
    table.getRange(`A${targetRow}:H${targetRow}`).format.fill = paleOrange;
  }
}

ablations.getRange("A1:L1").values = [[
  "Ablation",
  "Display name",
  "Effect definition",
  "Estimated pass rate",
  "Difference",
  "CI lower",
  "CI upper",
  "Raw p",
  "Holm p",
  "Cases",
  "Bootstrap",
  "Permutations",
]];
ablations.getRange("A2:L5").values = payload.ablations.map((row) => [
  row.ablation,
  row.display_name,
  row.effect_definition,
  row.estimated_pass_rate,
  row.difference,
  row.ci_lower,
  row.ci_upper,
  row.p_value,
  row.p_holm,
  row.n_cases,
  row.n_bootstrap,
  row.n_permutation,
]);
ablations.getRange("A1:L5").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
ablations.getRange("A1:L1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
ablations.getRange("D2:I5").format.numberFormat = "0.000000";
ablations.getRange("A1:L7").format.columnWidth = 20;
ablations.freezePanes.freezeRows(1);
ablations.tables.add("A1:L5", true, "Table4Ablations");

table.mergeCells("A28:H28");
table.getRange("A28").values = [[
  "E. Frozen fault handling: B4 minus B2",
]];
table.getRange("A28:H28").format = {
  fill: paleGray,
  font: { name: "Arial", size: 10, bold: true, color: navy },
  borders: {
    top: { style: "medium", color: navy },
    bottom: { style: "thin", color: rule },
  },
};
table.getRange("A29:H29").values = [[
  "Fault type",
  "Failure detection difference, pp",
  "95% CI, pp",
  "Correct terminal difference, pp",
  "95% CI, pp",
  "Cases",
  "Runs per system",
  "Scoring note",
]];
table.getRange("A29:H29").format = {
  fill: blue,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: navy } },
};
table.getRange("A29").format.horizontalAlignment = "left";
table.getRange("H29").format.horizontalAlignment = "left";
table.getRange("A29:H29").format.rowHeight = 38;
table.getRange("A30:H37").values = payload.faults.map((row) => [
  row.fault_label,
  "",
  "",
  "",
  "",
  row.n_cases,
  row.n_cases * 3,
  row.scoring_note,
]);
table.getRange("A30:H37").format = {
  font: { name: "Arial", size: 8, color: text },
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
table.getRange("B30:G37").format.horizontalAlignment = "center";
table.getRange("A30:H37").format.rowHeight = 30;
for (let index = 0; index < 8; index += 1) {
  const targetRow = index + 30;
  const sourceRow = index + 2;
  table.getRange(`B${targetRow}`).formulas = [[
    `=TEXT('Fault Handling'!D${sourceRow}*100,"+0.0;-0.0;0.0")&" pp"`,
  ]];
  table.getRange(`C${targetRow}`).formulas = [[
    `=TEXT('Fault Handling'!E${sourceRow}*100,"+0.0;-0.0;0.0")&" to "&TEXT('Fault Handling'!F${sourceRow}*100,"+0.0;-0.0;0.0")`,
  ]];
  table.getRange(`D${targetRow}`).formulas = [[
    `=TEXT('Fault Handling'!G${sourceRow}*100,"+0.0;-0.0;0.0")&" pp"`,
  ]];
  table.getRange(`E${targetRow}`).formulas = [[
    `=TEXT('Fault Handling'!H${sourceRow}*100,"+0.0;-0.0;0.0")&" to "&TEXT('Fault Handling'!I${sourceRow}*100,"+0.0;-0.0;0.0")`,
  ]];
}
table.getRange("A37:H37").format.fill = paleOrange;
table.getRange("A37:H37").format.rowHeight = 48;

faults.getRange("A1:J1").values = [[
  "Fault type",
  "Fault label",
  "Cases",
  "Detection difference",
  "Detection CI lower",
  "Detection CI upper",
  "Terminal difference",
  "Terminal CI lower",
  "Terminal CI upper",
  "Scoring note",
]];
faults.getRange("A2:J9").values = payload.faults.map((row) => [
  row.fault_type,
  row.fault_label,
  row.n_cases,
  row.detection_difference,
  row.detection_ci_lower,
  row.detection_ci_upper,
  row.terminal_difference,
  row.terminal_ci_lower,
  row.terminal_ci_upper,
  row.scoring_note,
]);
faults.getRange("A1:J9").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
faults.getRange("A1:J1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
faults.getRange("D2:I9").format.numberFormat = "0.000000";
faults.getRange("A1:I11").format.columnWidth = 20;
faults.getRange("J1:J11").format.columnWidth = 68;
faults.freezePanes.freezeRows(1);
faults.tables.add("A1:J9", true, "Table4FaultHandling");

const notes = [
  "The primary endpoint used patient-clustered bootstrap 95% CIs (2,000 resamples) and a two-sided paired sign-permutation test (100,000 draws).",
  "Ablation p values are Holm-adjusted across four paired comparisons. Traceability and repeatability endpoints are descriptive; no p values were prespecified.",
  "Exact extractive support and passage-citation validity are automated contract checks, not expert biomedical factuality or semantic retrieval assessments.",
  "Planning errors indicate action-specification noncompliance, not disagreement with survival labels. The strict post-hoc report-contract audit changed 0/600 clean outcomes.",
  "The internal deterministic verifier is B4-only and is N/A for B2; it is not reported as a zero-valued comparator metric.",
  "Unsupported request*: B4 detected the request and safely exited, but the frozen terminal rule scored the exit as unsuccessful; no post-hoc correction was made.",
  "All results are technical Agent-benchmark endpoints and do not establish clinical utility, diagnostic accuracy, treatment benefit, deployment readiness, or patient outcomes.",
];
for (let index = 0; index < notes.length; index += 1) {
  const row = index + 39;
  table.mergeCells(`A${row}:H${row}`);
  table.getRange(`A${row}`).values = [[notes[index]]];
}
table.getRange("A39:H45").format = {
  font: { name: "Arial", size: 8, color: midGray },
  wrapText: true,
  verticalAlignment: "center",
};
table.getRange("A39:H45").format.rowHeight = 24;
table.getRange("A39:H39").format.borders = {
  top: { style: "medium", color: navy },
};

table.getRange("A1:A48").format.columnWidth = 31;
table.getRange("B1:B48").format.columnWidth = 31;
table.getRange("C1:F48").format.columnWidth = 23;
table.getRange("G1:G48").format.columnWidth = 23;
table.getRange("H1:H48").format.columnWidth = 42;
table.freezePanes.freezeRows(5);

corrections.getRange("A1:E1").values = [[
  "Original name",
  "Revised name",
  "Reference",
  "Status",
  "Reason",
]];
corrections.getRange("A2:E6").values = payload.corrections.map((row) => [
  row.original_name,
  row.revised_name,
  row.reference,
  row.status,
  row.reason,
]);
corrections.getRange("A1:E6").format = {
  font: { name: "Arial", size: 8, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
corrections.getRange("A1:E1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
corrections.getRange("A1:E8").format.columnWidth = 42;
corrections.freezePanes.freezeRows(1);
corrections.tables.add("A1:E6", true, "Table4MetricAudit");

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
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
provenance.getRange("A1:C1").format = {
  fill: navy,
  font: { name: "Arial", size: 8, bold: true, color: "#FFFFFF" },
};
provenance.getRange("A1:A20").format.columnWidth = 72;
provenance.getRange("B1:B20").format.columnWidth = 50;
provenance.getRange("C1:C20").format.columnWidth = 68;
provenance.freezePanes.freezeRows(1);
provenance.tables.add(
  `A1:C${provenanceRows.length + 1}`,
  true,
  "Table4Provenance",
);

const definitionRows = [
  ["Term", "Definition"],
  ["B2", "Tool-using single LLM controller with frozen prognostic-model tools and assigned evidence passages, without the B4 verifier-repair loop."],
  ["B4", "Role-specialised, verifier-guided closed loop with persistent structured state, conditional replanning, tool retry, and one synthesis revision."],
  ["Frozen independently implemented composite pass", "Prespecified deterministic technical endpoint retained unchanged after the offline audit."],
  ["Exact extractive support", "A generated claim exactly matches a sentence in its assigned cited passage."],
  ["Assigned-passage citation-ID validity", "Every citation identifier belongs to the assigned passage set."],
  ["Exact three-run agreement", "A case has the same binary composite result in all three repeated runs."],
  ["Planning error", "Initial or final action specification violates the frozen plan/tool contract; unrelated to survival-label correctness."],
  ["Correct terminal outcome", "The system reaches the terminal state required by the frozen fault-specific scoring contract."],
  ["N/A verifier", "Systems without an internal verifier do not receive a zero verifier score; the process diagnostic is not comparable."],
];
definitions.getRange(`A1:B${definitionRows.length}`).values = definitionRows;
definitions.getRange(`A1:B${definitionRows.length}`).format = {
  font: { name: "Arial", size: 9, color: text },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: rule } },
};
definitions.getRange("A1:B1").format = {
  fill: navy,
  font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
};
definitions.getRange("A1:A20").format.columnWidth = 33;
definitions.getRange("B1:B20").format.columnWidth = 105;
definitions.getRange("A2:B20").format.rowHeight = 32;
definitions.freezePanes.freezeRows(1);

const preview = await workbook.render({
  sheetName: "Table 4",
  range: "A1:H45",
  scale: 2,
  format: "png",
});
await fs.writeFile(PREVIEW, new Uint8Array(await preview.arrayBuffer()));

const inspect = await workbook.inspect({
  kind: "table",
  range: "Table 4!A1:H45",
  include: "values,formulas",
  tableMaxRows: 50,
  tableMaxCols: 10,
});
await fs.writeFile(
  path.join(ROOT, "TABLE4_INSPECT.ndjson"),
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
  path.join(ROOT, "TABLE4_FORMULA_ERROR_SCAN.ndjson"),
  formulaErrors.ndjson,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUTPUT);

console.log(JSON.stringify({
  status: "TABLE4_WORKBOOK_CREATED",
  output: OUTPUT,
  preview: PREVIEW,
  recordCount: payload.source_gate.record_count,
  primaryDifference: payload.paired.difference,
}, null, 2));
