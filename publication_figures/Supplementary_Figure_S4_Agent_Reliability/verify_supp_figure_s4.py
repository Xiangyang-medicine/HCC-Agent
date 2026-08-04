from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent


def main() -> None:
    source_gate = json.loads((ROOT / "SOURCE_DATA_GATE.json").read_text(encoding="utf-8"))
    required = [
        ROOT / "Supplementary_Figure_S4_Agent_Reliability.svg",
        ROOT / "Supplementary_Figure_S4_Agent_Reliability.pdf",
        ROOT / "Supplementary_Figure_S4_Agent_Reliability.png",
        ROOT / "Supplementary_Figure_S4_Agent_Reliability.tiff",
    ]
    png = Image.open(required[2])
    tiff = Image.open(required[3])
    dpi = tuple(float(x) for x in tiff.info.get("dpi", (0, 0)))
    flow = pd.read_csv(ROOT / "panel_d_verification_repair_flow/source_data.csv")
    checks = {
        "source_data_gate_passed": source_gate.get("status") == "PASS",
        "formal_records_equal_4860": source_gate.get("formal_record_count") == 4860,
        "api_error_count_zero": source_gate.get("api_error_count") == 0,
        "paired_cases_equal_100": source_gate.get("paired_cases") == 100,
        "clean_runs_equal_300_per_system": (
            source_gate.get("clean_b2_runs") == 300 and source_gate.get("clean_b4_runs") == 300
        ),
        "b4_flow_totals_equal_300": int(flow["n_runs"].sum()) == 300,
        "all_outputs_exist": all(path.exists() and path.stat().st_size > 0 for path in required),
        "combined_png_at_least_1800px_wide": png.width >= 1800,
        "tiff_dpi_at_least_600": min(dpi) >= 599,
        "editable_svg_text_preserved": "font" in required[0].read_text(encoding="utf-8", errors="ignore"),
        "clinical_utility_claim_not_permitted": source_gate.get("clinical_utility_claim_permitted") is False,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "combined_png_pixels": [png.width, png.height],
        "tiff_dpi": list(dpi),
    }
    (ROOT / "SUPP_FIGURE_S4_QA_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
