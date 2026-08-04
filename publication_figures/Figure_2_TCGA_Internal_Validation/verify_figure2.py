from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "Figure_2_TCGA_Internal_Validation"


def check(name: str, condition: bool, detail: str, checks: list[dict]) -> None:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})


def finite_frame(path: Path) -> bool:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include="number")
    return bool(numeric.map(math.isfinite).all().all())


def main() -> None:
    checks: list[dict] = []
    manifest = json.loads((ROOT / "analysis_manifest.json").read_text(encoding="utf-8"))

    inputs_ok = all(
        Path(path).exists() for path in manifest["input_files"]
    )
    check("canonical_inputs_exist", inputs_ok, "all manifest inputs exist", checks)
    check(
        "canonical_v6_comparison_used",
        any("model_comparisons_v6.csv" in path for path in manifest["input_files"]),
        "model_comparisons_v6.csv is present in the manifest",
        checks,
    )
    check(
        "superseded_v5_not_used",
        not any("model_comparisons_v5" in path for path in manifest["input_files"]),
        "no v5 comparison file appears in the manifest",
        checks,
    )

    a = pd.read_csv(ROOT / "panel_a_model_discrimination" / "source_data.csv")
    check("panel_a_row_count", len(a) == 250, f"rows={len(a)}, expected=250", checks)
    check(
        "panel_a_fold_coverage",
        a.groupby(["model_short", "metric"]).size().eq(25).all(),
        "25 folds for every model and metric",
        checks,
    )

    b = pd.read_csv(ROOT / "panel_b_paired_differences" / "source_data.csv")
    check("panel_b_row_count", len(b) == 8, f"rows={len(b)}, expected=8", checks)
    m4_h = b.loc[
        b["comparison_label"].eq("M4 vs M1")
        & b["metric_label"].eq("Harrell C")
    ].iloc[0]
    m4_u = b.loc[
        b["comparison_label"].eq("M4 vs M1")
        & b["metric_label"].eq("Uno C")
    ].iloc[0]
    check(
        "panel_b_m4_harrell_adjusted_p",
        abs(float(m4_h["p_value_adjusted"]) - 0.0959040959040959) < 1e-12,
        f"adjusted P={float(m4_h['p_value_adjusted']):.15f}",
        checks,
    )
    check(
        "panel_b_m4_uno_adjusted_p",
        abs(float(m4_u["p_value_adjusted"]) - 1.0) < 1e-12,
        f"adjusted P={float(m4_u['p_value_adjusted']):.3f}",
        checks,
    )
    check(
        "panel_b_no_false_superiority",
        not bool(m4_h["significant_adjusted"])
        and not bool(m4_u["significant_adjusted"]),
        "M4 vs M1 is non-significant after correction for both metrics",
        checks,
    )

    c = pd.read_csv(ROOT / "panel_c_oof_survival" / "source_data_patients.csv")
    stats = json.loads(
        (ROOT / "panel_c_oof_survival" / "statistics.json").read_text(encoding="utf-8")
    )
    check("panel_c_patient_count", len(c) == 363, f"patients={len(c)}", checks)
    check(
        "panel_c_five_oof_predictions",
        c["oof_repeats"].eq(5).all(),
        "every patient has five OOF predictions",
        checks,
    )
    check(
        "panel_c_group_total",
        stats["n_high"] + stats["n_low"] == 363,
        f"high={stats['n_high']}, low={stats['n_low']}",
        checks,
    )
    check(
        "panel_c_hr_direction",
        stats["cox_hr_high_vs_low"] > 1,
        f"HR={stats['cox_hr_high_vs_low']:.4f}",
        checks,
    )

    d = pd.read_csv(ROOT / "panel_d_prediction_error" / "source_data.csv")
    check("panel_d_row_count", len(d) == 150, f"rows={len(d)}, expected=150", checks)
    check(
        "panel_d_fold_coverage",
        d.groupby(["model_short", "horizon_months"]).size().eq(25).all(),
        "25 folds for each model/horizon combination",
        checks,
    )

    e = pd.read_csv(ROOT / "panel_e_sensitivity" / "source_data.csv")
    check("panel_e_row_count", len(e) == 45, f"rows={len(e)}, expected=45", checks)
    m4 = e.loc[e["model_short"].eq("M4")]
    check(
        "panel_e_m4_harrell_rank",
        m4.loc[m4["metric"].eq("Harrell C"), "rank"].eq(1).all(),
        "M4 Harrell C rank is 1 in SA1-SA3",
        checks,
    )
    check(
        "panel_e_m4_ibs_rank",
        m4.loc[m4["metric"].eq("IBS"), "rank"].eq(1).all(),
        "M4 IBS rank is 1 in SA1-SA3",
        checks,
    )

    source_files = [
        ROOT / "panel_a_model_discrimination" / "source_data.csv",
        ROOT / "panel_b_paired_differences" / "source_data.csv",
        ROOT / "panel_c_oof_survival" / "source_data_patients.csv",
        ROOT / "panel_c_oof_survival" / "source_data_km_curves.csv",
        ROOT / "panel_c_oof_survival" / "source_data_at_risk.csv",
        ROOT / "panel_d_prediction_error" / "source_data.csv",
        ROOT / "panel_e_sensitivity" / "source_data.csv",
    ]
    check(
        "all_source_data_finite",
        all(finite_frame(path) for path in source_files),
        "all numeric source-data cells are finite",
        checks,
    )

    exports = [STEM.with_suffix(suffix) for suffix in [".svg", ".pdf", ".png", ".tiff"]]
    check(
        "assembled_exports_exist",
        all(path.exists() and path.stat().st_size > 10_000 for path in exports),
        ", ".join(f"{path.suffix}={path.stat().st_size if path.exists() else 0}" for path in exports),
        checks,
    )
    svg_text = STEM.with_suffix(".svg").read_text(encoding="utf-8")
    check(
        "svg_text_editable",
        "<text" in svg_text and "font-family" in svg_text,
        "SVG contains editable text elements",
        checks,
    )
    pdf = PdfReader(str(STEM.with_suffix(".pdf")))
    check("pdf_single_page", len(pdf.pages) == 1, f"pages={len(pdf.pages)}", checks)
    tiff = Image.open(STEM.with_suffix(".tiff"))
    width, height = tiff.size
    check(
        "tiff_resolution",
        width >= 4300 and height >= 4800,
        f"pixels={width}x{height}",
        checks,
    )

    panel_exports_ok = True
    for panel_dir in ROOT.glob("panel_*"):
        stem = panel_dir / panel_dir.name
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            path = stem.with_suffix(suffix)
            panel_exports_ok = panel_exports_ok and path.exists() and path.stat().st_size > 2_000
    check(
        "standalone_panel_exports_exist",
        panel_exports_ok,
        "all five panels have SVG/PDF/PNG/TIFF exports",
        checks,
    )

    success = all(item["passed"] for item in checks)
    report = {
        "figure": "Figure_2_TCGA_Internal_Validation",
        "status": "FIGURE2_QA_PASSED" if success else "FIGURE2_QA_FAILED",
        "success": success,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    (ROOT / "FIGURE2_QA_GATE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
