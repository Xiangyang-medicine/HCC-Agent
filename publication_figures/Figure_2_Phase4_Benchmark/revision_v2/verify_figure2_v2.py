from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, bool] = {}
    required = [
        ROOT / "Figure_2_Phase4_Benchmark_v2.svg",
        ROOT / "Figure_2_Phase4_Benchmark_v2.pdf",
        ROOT / "Figure_2_Phase4_Benchmark_v2.tiff",
        ROOT / "Figure_2_Phase4_Benchmark_v2.png",
        ROOT / "methodological_audit" / "AUDIT_GATE.json",
        ROOT / "methodological_audit" / "OFFLINE_SCORING_AUDIT.md",
        ROOT / "panel_b_primary_endpoint" / "source_data.csv",
        ROOT / "panel_c_ablation_effects" / "source_data.csv",
        ROOT / "panel_d_traceability_reliability" / "source_data.csv",
        ROOT / "panel_e_fault_handling" / "source_data.csv",
    ]
    for path in required:
        checks[f"exists_{path.name}"] = path.exists() and path.stat().st_size > 0

    audit_gate = json.loads(
        (ROOT / "methodological_audit" / "AUDIT_GATE.json").read_text(
            encoding="utf-8"
        )
    )
    checks["audit_status"] = (
        audit_gate.get("status") == "POSTHOC_OFFLINE_AUDIT_COMPLETED"
    )
    checks["raw_records_4860"] = audit_gate.get("raw_record_count") == 4860
    checks["confirmatory_endpoint_unchanged"] = (
        audit_gate.get("confirmatory_endpoint_modified") is False
    )
    checks["strict_audit_no_clean_disagreement"] = (
        audit_gate.get("strict_vs_frozen_clean_disagreement_count") == 0
    )

    primary = pd.read_csv(
        ROOT / "panel_b_primary_endpoint" / "paired_comparison.csv"
    ).iloc[0]
    checks["primary_difference_13pp"] = np.isclose(primary["difference"], 0.13)
    checks["primary_paired_runs_300"] = int(primary["n_paired_runs"]) == 300
    checks["primary_pair_counts_sum_300"] = (
        int(primary["both_pass"])
        + int(primary["b4_only_pass"])
        + int(primary["b2_only_pass"])
        + int(primary["both_fail"])
        == 300
    )

    panel_d = pd.read_csv(
        ROOT / "panel_d_traceability_reliability" / "source_data.csv"
    )
    checks["no_schema_metric_in_main_panels"] = not panel_d["metric"].eq(
        "schema_valid"
    ).any()
    checks["no_verifier_zero_metric_in_main_panels"] = not panel_d["metric"].eq(
        "external_verifier_passed"
    ).any()

    for relative in [
        "panel_b_primary_endpoint/source_data.csv",
        "panel_c_ablation_effects/source_data.csv",
        "panel_d_traceability_reliability/source_data.csv",
        "panel_e_fault_handling/source_data.csv",
    ]:
        frame = pd.read_csv(ROOT / relative)
        numeric = frame.select_dtypes(include=[np.number])
        checks[f"no_inf_{relative}"] = not np.isinf(numeric.to_numpy()).any()

    svg_text = (ROOT / "Figure_2_Phase4_Benchmark_v2.svg").read_text(
        encoding="utf-8"
    )
    checks["svg_editable_text"] = "<text" in svg_text
    checks["svg_has_five_panel_labels"] = all(
        re.search(rf">\s*{letter}\s*</text>", svg_text)
        for letter in "abcde"
    )

    with Image.open(ROOT / "Figure_2_Phase4_Benchmark_v2.tiff") as image:
        checks["tiff_minimum_dimensions"] = (
            image.width >= 4000 and image.height >= 3500
        )
        dpi = image.info.get("dpi", (0, 0))
        checks["tiff_600dpi"] = round(float(dpi[0])) == 600
    with Image.open(ROOT / "Figure_2_Phase4_Benchmark_v2.png") as image:
        dpi = image.info.get("dpi", (0, 0))
        checks["png_300dpi"] = round(float(dpi[0])) == 300

    # Pandas and NumPy predicates may return np.bool_; normalize all fields so
    # the gate contains native JSON booleans rather than strings or scalars.
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    gate = {
        "status": "FIGURE2_V2_QA_PASSED" if not failed else "FIGURE2_V2_QA_FAILED",
        "success": not failed,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in required[:4]
            if path.exists()
        },
    }
    (ROOT / "FIGURE2_V2_QA_GATE.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
