from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "Supplementary_Figure_S1_Time_Dependent_Calibration"
PANELS = {
    "a": ROOT / "panel_a_auc_trajectories" / "panel_a_auc_trajectories",
    "b": ROOT / "panel_b_calibration_12m" / "panel_b_calibration_12m",
    "c": ROOT / "panel_c_calibration_36m" / "panel_c_calibration_36m",
    "d": ROOT / "panel_d_calibration_60m" / "panel_d_calibration_60m",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    source_gate = json.loads((ROOT / "SOURCE_DATA_GATE.json").read_text(encoding="utf-8"))
    auc = pd.read_csv(PANELS["a"].parent / "source_data.csv")
    calibration = [pd.read_csv(PANELS[key].parent / "source_data.csv") for key in ["b", "c", "d"]]
    with Image.open(STEM.with_suffix(".png")) as image:
        png_size = tuple(int(value) for value in image.size)
    with Image.open(STEM.with_suffix(".tiff")) as image:
        raw_dpi = image.info.get("dpi")
        tiff_dpi = tuple(float(value) for value in raw_dpi) if raw_dpi else None
    svg = STEM.with_suffix(".svg").read_text(encoding="utf-8", errors="replace")
    checks = {
        "source_gate_success": source_gate["success"] is True,
        "auc_375_rows": len(auc) == 375,
        "calibration_60_rows_each": all(len(frame) == 60 for frame in calibration),
        "six_bins_each_model_repeat": all(bool((frame.groupby(["model_short", "repeat"]).size() == 6).all()) for frame in calibration),
        "calibration_bounded": all(bool(frame["observed_event_probability_km"].between(0, 1).all() and frame["mean_predicted_event_risk"].between(0, 1).all()) for frame in calibration),
        "combined_exports": all(STEM.with_suffix(ext).is_file() for ext in [".svg", ".pdf", ".png", ".tiff"]),
        "panel_exports": all(stem.with_suffix(ext).is_file() for stem in PANELS.values() for ext in [".svg", ".pdf", ".png", ".tiff"]),
        "editable_svg_text": "<text" in svg and "font-family" in svg,
        "png_large_enough": png_size[0] >= 2000 and png_size[1] >= 1300,
        "tiff_600_dpi": tiff_dpi is not None and min(tiff_dpi) >= 590,
    }
    success = all(checks.values())
    report = {
        "status": "SUPP_FIGURE_S1_QA_PASSED" if success else "SUPP_FIGURE_S1_QA_FAILED",
        "success": bool(success),
        "checks": checks,
        "files": {ext.lstrip("."): {"path": str(STEM.with_suffix(ext)), "sha256": sha256(STEM.with_suffix(ext))} for ext in [".svg", ".pdf", ".png", ".tiff"]},
        "image_size_px": png_size,
        "tiff_dpi": tiff_dpi,
    }
    (ROOT / "SUPP_FIGURE_S1_QA_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
