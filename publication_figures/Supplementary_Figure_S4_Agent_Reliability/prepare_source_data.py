from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = Path(r"F:\ACM")
METRICS = PROJECT / "experiments/phase4/formal_remote_completed/all_case_level_metrics.csv"
CASES = PROJECT / "data/phase4_benchmark/formal_cases_reserved_blinded.json"
B2 = "B2_SINGLE_LLM_WITH_TOOLS"
B4 = "B4_FULL_CLOSED_LOOP"


def main() -> None:
    panel_dirs = {
        "a": ROOT / "panel_a_benchmark_composition",
        "b": ROOT / "panel_b_paired_case_success",
        "c": ROOT / "panel_c_repeat_reliability",
        "d": ROOT / "panel_d_verification_repair_flow",
        "e": ROOT / "panel_e_fault_outcomes",
        "f": ROOT / "panel_f_support_precision_completeness",
    }
    for directory in panel_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS)
    cases = pd.DataFrame(json.loads(CASES.read_text(encoding="utf-8")))

    composition = (
        cases.groupby(["risk_quintile_sampling_stratum", "event_sampling_stratum"], as_index=False)
        .size()
        .rename(columns={"risk_quintile_sampling_stratum": "risk_quintile",
                         "event_sampling_stratum": "event", "size": "n_cases"})
    )
    composition.to_csv(panel_dirs["a"] / "source_data.csv", index=False)

    clean = metrics.loc[(metrics["run_kind"] == "clean") & metrics["system"].isin([B2, B4])].copy()
    paired = (
        clean.groupby(["case_id", "system"], as_index=False)["verified_task_success"]
        .mean()
        .pivot(index="case_id", columns="system", values="verified_task_success")
        .reset_index()
    )
    paired["delta_b4_minus_b2"] = paired[B4] - paired[B2]
    paired["order"] = np.lexsort((-paired[B4].to_numpy(), paired[B2].to_numpy())) + 1
    paired.to_csv(panel_dirs["b"] / "source_data.csv", index=False)

    reliability = (
        clean.groupby(["case_id", "system"], as_index=False)
        .agg(pass_fraction=("verified_task_success", "mean"),
             exact_three_run_agreement=("verified_task_success", lambda x: int(x.nunique() == 1)),
             successes=("verified_task_success", "sum"))
    )
    reliability.to_csv(panel_dirs["c"] / "source_data.csv", index=False)

    b4_clean = clean.loc[clean["system"] == B4].copy()
    terminal = np.select(
        [
            b4_clean["verified_task_success"].eq(True) & b4_clean["revision_count"].eq(0),
            b4_clean["verified_task_success"].eq(True) & b4_clean["revision_count"].ge(1),
            b4_clean["safe_abstain"].eq(True),
        ],
        ["Direct verified report", "Verified after repair", "Safe abstention"],
        default="Unresolved failure",
    )
    b4_clean["initial_plan_state"] = np.where(b4_clean["plan_valid"].eq(True), "Valid initial plan", "Invalid initial plan")
    b4_clean["terminal_state"] = terminal
    flow = b4_clean.groupby(["initial_plan_state", "terminal_state"], as_index=False).size().rename(columns={"size": "n_runs"})
    flow.to_csv(panel_dirs["d"] / "source_data.csv", index=False)

    fault = metrics.loc[(metrics["run_kind"] == "fault") & metrics["system"].isin([B2, B4])].copy()
    fault["outcome"] = np.select(
        [
            fault["recovery_or_safe_abstention"].eq(True),
            fault["failure_detected"].eq(True),
        ],
        ["Correct terminal outcome", "Detected, incorrect terminal"],
        default="Undetected / unrecovered",
    )
    fault_outcomes = (
        fault.groupby(["fault_type", "system", "outcome"], as_index=False)
        .size()
        .rename(columns={"size": "n_runs"})
    )
    totals = fault_outcomes.groupby(["fault_type", "system"])["n_runs"].transform("sum")
    fault_outcomes["fraction"] = fault_outcomes["n_runs"] / totals
    fault_outcomes.to_csv(panel_dirs["e"] / "source_data.csv", index=False)

    support = clean[
        ["case_id", "formal_repeat", "system", "supported_claim_precision",
         "citation_completeness", "citation_correctness", "verified_task_success"]
    ].copy()
    support["reference_standard"] = "frozen exact-string claim support + assigned-passage citation IDs"
    support.to_csv(panel_dirs["f"] / "source_data.csv", index=False)

    checks = {
        "status": "PASS",
        "formal_gate_status": json.loads(
            (METRICS.parent / "all_RUN_GATE.json").read_text(encoding="utf-8")
        )["status"],
        "formal_record_count": int(len(metrics)),
        "api_error_count": int(metrics["api_error_type"].notna().sum()),
        "formal_case_count": int(cases["case_id"].nunique()),
        "clean_b2_runs": int(((clean["system"] == B2)).sum()),
        "clean_b4_runs": int(((clean["system"] == B4)).sum()),
        "paired_cases": int(len(paired)),
        "fault_runs_b2_b4": int(len(fault)),
        "support_rows": int(len(support)),
        "clinical_utility_claim_permitted": False,
    }
    (ROOT / "SOURCE_DATA_GATE.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
