from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent


def main() -> None:
    source_gate = json.loads((ROOT / "SOURCE_DATA_GATE.json").read_text(encoding="utf-8"))
    required = [
        ROOT / "Supplementary_Figure_S3_Cross_Platform_Input_QC.svg",
        ROOT / "Supplementary_Figure_S3_Cross_Platform_Input_QC.pdf",
        ROOT / "Supplementary_Figure_S3_Cross_Platform_Input_QC.png",
        ROOT / "Supplementary_Figure_S3_Cross_Platform_Input_QC.tiff",
    ]
    png = Image.open(required[2])
    tiff = Image.open(required[3])
    gpl571_rows = pd.read_csv(ROOT / "panel_a_cohort_flow/source_data.csv")
    gpl571_rows = gpl571_rows.loc[gpl571_rows["cohort"].eq("GSE14520 · GPL571")]
    checks = {
        "source_data_gate_passed": source_gate.get("status") == "PASS",
        "all_outputs_exist": all(path.exists() and path.stat().st_size > 0 for path in required),
        "combined_png_at_least_1800px_wide": png.width >= 1800,
        "tiff_dpi_at_least_600": min(tiff.info.get("dpi", (0, 0))) >= 599,
        "editable_svg_text_preserved": "font" in required[0].read_text(encoding="utf-8", errors="ignore"),
        "gpl571_only_marked_excluded": (
            set(gpl571_rows["decision"]) == {"screened", "excluded_insufficient_n"}
            and gpl571_rows.sort_values("stage_order").iloc[-1]["decision"] == "excluded_insufficient_n"
            and "included" not in set(gpl571_rows["decision"])
        ),
        "gpl571_performance_not_reported": source_gate.get("gpl571_performance_rows_in_figure") == 0,
        "included_score_rows_equal_285": source_gate.get("score_rows") == 285,
    }
    dpi = tuple(float(x) for x in tiff.info.get("dpi", (0, 0)))
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "combined_png_pixels": [png.width, png.height],
        "tiff_dpi": list(dpi),
    }
    (ROOT / "SUPP_FIGURE_S3_QA_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
