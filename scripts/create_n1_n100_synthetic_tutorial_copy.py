"""Create an explicitly labelled N1-N100 tutorial copy of synthetic CSVs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(r"F:\ACM")
SOURCE = ROOT / "Outside data"
OUTPUT = SOURCE / "synthetic_N1_N100_DO_NOT_USE"
LABEL = "SYNTHETIC_DO_NOT_USE"


def new_record_id(row: pd.Series) -> str:
    identity = (
        row["case_id"],
        int(row["formal_repeat"]),
        row["run_kind"],
        row["system"],
        None if pd.isna(row["fault_type"]) or row["fault_type"] == "" else row["fault_type"],
    )
    return hashlib.sha256("|".join(map(str, identity)).encode()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_path = SOURCE / "Case_level_metrics_4860rows.csv"
    frame = pd.read_csv(case_path)
    original_ids = frame["record_id"].copy()
    frame["case_id"] = frame["case_id"].str.extract(r"(\d+)$")[0].astype(int).map(
        lambda value: f"N{value}"
    )
    frame.insert(0, "data_status", LABEL)
    frame.insert(2, "original_synthetic_record_id", original_ids)
    frame["record_id"] = frame.apply(new_record_id, axis=1)
    assert len(frame) == 4860
    assert frame["case_id"].nunique() == 100
    assert set(frame["case_id"]) == {f"N{index}" for index in range(1, 101)}
    assert frame["record_id"].nunique() == 4860
    frame.to_csv(OUTPUT / "SYNTHETIC_Case_level_metrics_4860rows_N1_N100.csv", index=False)

    for name in [
        "Clean_summary.csv",
        "Ablation_summary.csv",
        "fault_summary.csv",
        "primary_comparison.csv",
    ]:
        summary = pd.read_csv(SOURCE / name)
        summary.insert(0, "data_status", LABEL)
        summary.to_csv(OUTPUT / f"SYNTHETIC_{name}", index=False)


if __name__ == "__main__":
    main()
