from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT = Path(r"F:\ACM")
FIG_ROOT = PROJECT / "publication_figures"
TAB_ROOT = PROJECT / "publication_tables"
OUT_JSON = TAB_ROOT / "MAIN_CHINESE_EXPLANATIONS_QA_GATE.json"
OUT_MD = TAB_ROOT / "MAIN_CHINESE_EXPLANATIONS_QA_REPORT.md"

DOCS = {
    "Figure 1": FIG_ROOT / "Figure_1_System_Architecture" / "中文详细解读.md",
    "Figure 2": FIG_ROOT / "Figure_2_TCGA_Internal_Validation" / "中文详细解读.md",
    "Figure 3": FIG_ROOT / "Figure_3_External_Transport" / "中文详细解读.md",
    "Figure 4": FIG_ROOT / "Figure_4_Agent_Benchmark" / "中文详细解读.md",
    "Table 1": TAB_ROOT / "Table_1_Cohort_Characteristics" / "中文详细解读.md",
    "Table 2": TAB_ROOT / "Table_2_Internal_Model_Performance" / "中文详细解读.md",
    "Table 3": TAB_ROOT / "Table_3_External_Transport" / "中文详细解读.md",
    "Table 4": TAB_ROOT / "Table_4_Agent_Benchmark" / "中文详细解读.md",
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def contains_all(text: str, terms: list[str]) -> bool:
    return all(term in text for term in terms)


def main() -> None:
    texts = {name: read(path) for name, path in DOCS.items() if path.exists()}
    checks: dict[str, bool] = {}

    checks["eight_explanation_documents_exist"] = len(texts) == 8
    checks["four_figure_explanations_exist"] = all(f"Figure {i}" in texts for i in range(1, 5))
    checks["four_table_explanations_exist"] = all(f"Table {i}" in texts for i in range(1, 5))
    checks["all_documents_are_detailed"] = all(len(text) >= 3500 for text in texts.values())
    checks["all_documents_have_required_structure"] = all(
        contains_all(
            text,
            [
                "为什么要做",
                "使用的数据",
                "分析方法",
                "对整篇论文的意义",
                "能说明什么、不能说明什么",
                "术语解释",
                "对应材料",
            ],
        )
        for text in texts.values()
    )
    checks["utf8_text_clean"] = all(
        "�" not in text and not any(token in text for token in ["鈥", "揗", "揝", "¡Á"])
        for text in texts.values()
    )
    checks["no_patch_markers"] = all(not re.search(r"(?m)^\+", text) for text in texts.values())
    checks["master_index_exists"] = (TAB_ROOT / "正图正表中文解读总索引.md").exists()

    figure1 = texts["Figure 1"]
    checks["figure1_missing_asset_is_transparently_disclosed"] = contains_all(
        figure1, ["最终Mermaid图件尚未保存", "待最终图件导出后", "不能证明系统性能优于基线"]
    )

    table1_source = pd.read_csv(
        TAB_ROOT
        / "Table_1_Cohort_Characteristics"
        / "source_data"
        / "cohort_characteristics_numeric.csv"
    )
    table1 = texts["Table 1"]
    checks["table1_primary_counts_match"] = contains_all(
        table1, ["363", "129", "221", "85", "64", "27", "21例", "15/15"]
    ) and len(table1_source) > 0

    table2_perf = pd.read_csv(
        TAB_ROOT
        / "Table_2_Internal_Model_Performance"
        / "Table_2_Internal_Model_Performance.csv"
    )
    table2_cmp = pd.read_csv(
        TAB_ROOT
        / "Table_2_Internal_Model_Performance"
        / "Table_2_Model_Comparisons.csv"
    )
    figure2 = texts["Figure 2"]
    table2 = texts["Table 2"]
    checks["figure2_values_match"] = contains_all(
        figure2, ["Harrell C 0.641", "Uno C 0.627", "HR=2.21", "9.50×10⁻⁶", "IBS为0.184"]
    )
    checks["table2_values_match"] = (
        len(table2_perf) == 5
        and len(table2_cmp) == 8
        and contains_all(table2, ["0.641", "0.627", "0.184", "+0.078", "校正P=0.096", "校正P=0.008"])
    )

    table3_perf = pd.read_csv(
        TAB_ROOT
        / "Table_3_External_Transport"
        / "Table_3_External_Transport_Performance.csv"
    )
    table3_threshold = pd.read_csv(
        TAB_ROOT
        / "Table_3_External_Transport"
        / "Table_3_Frozen_Threshold_Stratification.csv"
    )
    figure3 = texts["Figure 3"]
    table3 = texts["Table 3"]
    checks["figure3_values_match"] = contains_all(
        figure3, ["0.629", "0.635", "0.603", "0.601", "HR=2.13", "HR=2.47", "P=0.057"]
    )
    checks["table3_values_match"] = (
        len(table3_perf) == 2
        and len(table3_threshold) == 2
        and contains_all(table3, ["0.629", "0.635", "0.603", "0.601", "48.0个月", "44.8个月", "−0.0100"])
    )

    table4_primary = pd.read_csv(
        TAB_ROOT
        / "Table_4_Agent_Benchmark"
        / "Table_4_Primary_Agent_Benchmark.csv"
    )
    table4_ablation = pd.read_csv(
        TAB_ROOT / "Table_4_Agent_Benchmark" / "Table_4_Ablation_Effects.csv"
    )
    table4_fault = pd.read_csv(
        TAB_ROOT / "Table_4_Agent_Benchmark" / "Table_4_Fault_Handling.csv"
    )
    figure4 = texts["Figure 4"]
    table4 = texts["Table 4"]
    checks["figure4_values_match"] = contains_all(
        figure4, ["94.7%", "81.7%", "+13.0个百分点", "0.00002", "4,860", "API错误：0"]
    )
    checks["table4_values_match"] = (
        len(table4_primary) == 2
        and len(table4_ablation) == 4
        and len(table4_fault) == 8
        and contains_all(table4, ["245/300", "284/300", "下降94.7", "下降16.7", "下降22.7", "4,860"])
    )

    checks["reporting_boundaries_present"] = (
        "没有证明M4显著优于M1" in figure2
        and "不是完整M4外部验证" in figure3
        and "不能说明B4达到医生水平" in figure4
        and "GPL571样本量过少，未进行分析" in figure3
        and "不是RNA测序加临床变量的完整M4外部验证" in table3
        and "不能说明B4达到或超过医生" in table4
    )
    checks["planning_error_definition_correct"] = (
        "规划错误不是预测标签错误" in figure4
        and "不是“风险高低与真实生存不一致”" in figure4
    )
    checks["source_and_code_references_present"] = all(
        "源数据" in text and "代码" in text for text in texts.values()
    )
    checks["existing_main_figure_assets_present"] = all(
        all(
            (
                FIG_ROOT
                / folder
                / f"{stem}.{suffix}"
            ).exists()
            for suffix in ["svg", "pdf", "png", "tiff"]
        )
        for folder, stem in [
            ("Figure_2_TCGA_Internal_Validation", "Figure_2_TCGA_Internal_Validation"),
            ("Figure_3_External_Transport", "Figure_3_External_Transport"),
            ("Figure_4_Agent_Benchmark", "Figure_4_Agent_Benchmark"),
        ]
    )
    checks["all_main_table_workbooks_present"] = all(
        any(folder.glob("*.xlsx"))
        for folder in [
            TAB_ROOT / "Table_1_Cohort_Characteristics",
            TAB_ROOT / "Table_2_Internal_Model_Performance",
            TAB_ROOT / "Table_3_External_Transport",
            TAB_ROOT / "Table_4_Agent_Benchmark",
        ]
    )

    checks = {name: bool(value) for name, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    failed = [name for name, value in checks.items() if not value]
    report = {
        "status": status,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": checks,
        "documents": {name: str(path) for name, path in DOCS.items()},
        "index": str(TAB_ROOT / "正图正表中文解读总索引.md"),
        "figure1_asset_status": "PENDING_FINAL_MERMAID_EXPORT",
        "reporting_boundaries": [
            "M4 has the strongest descriptive internal profile, but superiority versus M1 is not established after correction.",
            "External microarray results assess frozen gene-only transport, not full M4 validation.",
            "GPL571 N=21 was excluded before performance analysis.",
            "The Agent benchmark supports technical reliability only, not clinical utility or physician equivalence.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 正图正表中文解读QA报告",
        "",
        f"总体状态：**{status}**",
        "",
        f"- 通过：{sum(checks.values())}/{len(checks)}",
        f"- 正图解读：{sum(name.startswith('Figure') for name in texts)}",
        f"- 主表解读：{sum(name.startswith('Table') for name in texts)}",
        "- Figure 1图件状态：等待最终Mermaid导出；解读稿已明确标注。",
        "",
        "## 未通过项目",
        "",
        *(failed if failed else ["无。"]),
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "PASS":
        raise SystemExit(json.dumps({"failed": failed}, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
