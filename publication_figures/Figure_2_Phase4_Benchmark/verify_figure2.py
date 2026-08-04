from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RAW_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "phase4"
    / "formal_remote_completed"
    / "all_case_level_metrics.csv"
)

BOOL_COLUMNS = [
    "verified_task_success",
    "plan_valid",
    "tool_order_exact",
    "schema_valid",
    "numeric_fidelity",
    "forbidden_claim_case",
    "external_verifier_passed",
    "safe_abstain",
    "failure_detected",
    "recovery_or_safe_abstention",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.map(
        {
            True: 1.0,
            False: 0.0,
            "True": 1.0,
            "False": 0.0,
            "true": 1.0,
            "false": 0.0,
            1: 1.0,
            0: 0.0,
        }
    )


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=0))


def main() -> None:
    checks: dict[str, bool] = {}
    raw = pd.read_csv(RAW_CSV)
    for column in BOOL_COLUMNS:
        raw[column] = normalize_bool(raw[column])

    checks["raw_record_count_4860"] = len(raw) == 4860
    checks["raw_record_ids_unique"] = raw["record_id"].nunique() == 4860
    checks["raw_api_error_count_zero"] = raw["api_error_type"].notna().sum() == 0
    checks["raw_run_kind_counts"] = (
        raw.groupby("run_kind").size().to_dict()
        == {"ablation": 1200, "clean": 1500, "fault": 2160}
    )

    clean = raw[raw["run_kind"].eq("clean")]
    panel_a = pd.read_csv(ROOT / "panel_a_primary_success" / "source_data.csv")
    for _, row in panel_a.iterrows():
        source = clean[clean["system"].eq(row["system"])]
        checks[f"panel_a_rate_{row['system']}"] = close(
            row["rate"], source["verified_task_success"].mean()
        )
        checks[f"panel_a_count_{row['system']}"] = int(row["n_runs"]) == len(source)

    panel_d = pd.read_csv(ROOT / "panel_d_ablation" / "source_data.csv")
    for _, row in panel_d.iterrows():
        if row["system"] == "B4_FULL_CLOSED_LOOP":
            source = clean[clean["system"].eq(row["system"])]
        else:
            source = raw[
                raw["run_kind"].eq("ablation") & raw["system"].eq(row["system"])
            ]
        checks[f"panel_d_rate_{row['system']}"] = close(
            row["rate"], source["verified_task_success"].mean()
        )

    panel_e = pd.read_csv(
        ROOT / "panel_e_reliability_efficiency" / "source_data.csv"
    )
    for _, row in panel_e.iterrows():
        source = clean[clean["system"].eq(row["system"])]
        pivot = source.pivot(
            index="case_id",
            columns="formal_repeat",
            values="verified_task_success",
        )
        agreement = pivot.nunique(axis=1).eq(1).mean()
        checks[f"panel_e_agreement_{row['system']}"] = close(
            row["exact_three_run_agreement"], agreement
        )
        checks[f"panel_e_latency_{row['system']}"] = close(
            row["median_latency_ms"], source["wall_latency_ms"].median()
        )

    panel_f = pd.read_csv(ROOT / "panel_f_fault_matrix" / "source_data.csv")
    for _, row in panel_f.iterrows():
        source = raw[
            raw["run_kind"].eq("fault")
            & raw["fault_type"].eq(row["fault_type"])
            & raw["system"].eq(row["system"])
        ]
        key = f"{row['fault_type']}_{row['system']}"
        checks[f"panel_f_detection_{key}"] = close(
            row["failure_detection_rate"], source["failure_detected"].mean()
        )
        checks[f"panel_f_outcome_{key}"] = close(
            row["correct_outcome_rate"],
            source["recovery_or_safe_abstention"].mean(),
        )
        checks[f"panel_f_count_{key}"] = (
            int(row["n_cases"]) == 30 and int(row["n_runs"]) == 90
        )

    expected_panel_rows = {
        "panel_a_primary_success/source_data.csv": 5,
        "panel_b_functional_decomposition/source_data.csv": 20,
        "panel_c_grounding_safety/source_data.csv": 12,
        "panel_d_ablation/source_data.csv": 5,
        "panel_e_reliability_efficiency/source_data.csv": 3,
        "panel_f_fault_matrix/source_data.csv": 24,
    }
    for relative, expected in expected_panel_rows.items():
        frame = pd.read_csv(ROOT / relative)
        checks[f"source_rows_{relative}"] = len(frame) == expected
        numeric = frame.select_dtypes(include=[np.number])
        checks[f"source_no_inf_{relative}"] = not np.isinf(numeric.to_numpy()).any()

    stem = ROOT / "Figure_2_Phase4_Benchmark"
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = stem.with_suffix(suffix)
        checks[f"assembled_{suffix[1:]}_exists"] = path.exists() and path.stat().st_size > 0

    svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
    checks["svg_contains_editable_text"] = len(re.findall(r"<text\b", svg)) >= 150

    with Image.open(stem.with_suffix(".tiff")) as image:
        dpi = image.info.get("dpi", (0, 0))
        checks["tiff_600_dpi"] = min(dpi) >= 599
        checks["tiff_dimensions_sufficient"] = (
            image.width >= 4200 and image.height >= 5200
        )

    reader = PdfReader(stem.with_suffix(".pdf"))
    checks["pdf_single_page"] = len(reader.pages) == 1
    page = reader.pages[0]
    width_inches = float(page.mediabox.width) / 72
    height_inches = float(page.mediabox.height) / 72
    checks["pdf_full_width"] = 6.9 <= width_inches <= 7.3
    checks["pdf_height_within_page"] = height_inches <= 9.2
    fonts = []
    resources = page.get("/Resources")
    if resources and "/Font" in resources:
        for reference in resources["/Font"].values():
            font = reference.get_object()
            fonts.append(str(font.get("/BaseFont", "")))
    checks["pdf_fonts_embedded_subset"] = bool(fonts) and all(
        "+" in font and "Arial" in font for font in fonts
    )

    current_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and "superseded_layout_drafts" not in path.parts
    ]
    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(current_files)
        if path.name != "FIGURE2_QA_GATE.json"
    }
    checks = {key: bool(value) for key, value in checks.items()}
    success = all(checks.values())
    payload = {
        "status": "FIGURE2_VERIFIED" if success else "FIGURE2_QA_FAILED",
        "success": success,
        "check_count": int(len(checks)),
        "passed_count": int(sum(checks.values())),
        "failed_checks": [key for key, value in checks.items() if not value],
        "checks": checks,
        "assembled_pdf_size_inches": [width_inches, height_inches],
        "source_and_output_sha256": hashes,
    }
    (ROOT / "FIGURE2_QA_GATE.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in payload if key != "checks" and key != "source_and_output_sha256"}, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
