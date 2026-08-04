from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "Supplementary_Figure_S2_OOF_Stability"
PANELS = {
    "a": ROOT / "panel_a_repeat_correlation" / "panel_a_repeat_correlation",
    "b": ROOT / "panel_b_patient_rank_dispersion" / "panel_b_patient_rank_dispersion",
    "c": ROOT / "panel_c_m4_quintile_transition" / "panel_c_m4_quintile_transition",
    "d": ROOT / "panel_d_m5_quintile_transition" / "panel_d_m5_quintile_transition",
    "e": ROOT / "panel_e_consensus_and_qc" / "panel_e_consensus_and_qc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    source_gate = json.loads((ROOT / "SOURCE_DATA_GATE.json").read_text(encoding="utf-8"))
    pair_data = pd.read_csv(PANELS["a"].parent / "source_data.csv")
    stability = pd.read_csv(PANELS["b"].parent / "source_data.csv")
    m4 = pd.read_csv(PANELS["c"].parent / "source_data.csv")
    m5 = pd.read_csv(PANELS["d"].parent / "source_data.csv")
    consensus = pd.read_csv(PANELS["e"].parent / "source_data.csv")
    qc = pd.read_csv(PANELS["e"].parent / "source_data_qc.csv")
    with Image.open(STEM.with_suffix(".png")) as image:
        png_size = tuple(int(value) for value in image.size)
    with Image.open(STEM.with_suffix(".tiff")) as image:
        raw_dpi = image.info.get("dpi")
        tiff_dpi = tuple(float(value) for value in raw_dpi) if raw_dpi else None
    svg_text = STEM.with_suffix(".svg").read_text(encoding="utf-8", errors="replace")
    checks = {
        "source_gate_success": source_gate["success"] is True,
        "pair_rows_50": len(pair_data) == 50,
        "ten_pairs_per_model": bool((pair_data.groupby("model_short").size() == 10).all()),
        "stability_rows_1815": len(stability) == 1815,
        "363_patients_per_model": bool((stability.groupby("model_short").size() == 363).all()),
        "m4_density_3630_rows": len(m4) == 3630,
        "m5_density_3630_rows": len(m5) == 3630,
        "density_values_bounded": bool(
            m4[["risk_percentile_a", "risk_percentile_b"]].apply(lambda column: column.between(0, 1).all()).all()
            and m5[["risk_percentile_a", "risk_percentile_b"]].apply(lambda column: column.between(0, 1).all()).all()
        ),
        "consensus_rows_25": len(consensus) == 25,
        "consensus_fraction_sums": bool(
            (consensus.groupby("model_short")["fraction"].sum().round(10) == 1).all()
        ),
        "all_prediction_qc_passed": bool(
            (qc["bounded_fraction"] == 1).all() and (qc["monotonic_fraction"] == 1).all()
        ),
        "all_combined_exports_exist": all(STEM.with_suffix(ext).is_file() for ext in [".svg", ".pdf", ".png", ".tiff"]),
        "all_panel_exports_exist": all(stem.with_suffix(ext).is_file() for stem in PANELS.values() for ext in [".svg", ".pdf", ".png", ".tiff"]),
        "svg_has_editable_text": "<text" in svg_text and "font-family" in svg_text,
        "png_large_enough": png_size[0] >= 2000 and png_size[1] >= 1600,
        "tiff_approximately_600_dpi": tiff_dpi is not None and min(tiff_dpi) >= 590,
    }
    success = all(checks.values())
    report = {
        "status": "SUPP_FIGURE_S2_QA_PASSED" if success else "SUPP_FIGURE_S2_QA_FAILED",
        "success": bool(success),
        "checks": checks,
        "files": {
            ext.lstrip("."): {
                "path": str(STEM.with_suffix(ext)),
                "sha256": sha256(STEM.with_suffix(ext)),
            }
            for ext in [".svg", ".pdf", ".png", ".tiff"]
        },
        "image_size_px": png_size,
        "tiff_dpi": tiff_dpi,
    }
    (ROOT / "SUPP_FIGURE_S2_QA_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
