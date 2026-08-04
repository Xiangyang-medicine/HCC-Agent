from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FIGURE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_ROOT.parents[1]
INPUT_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "phase4"
    / "formal_remote_completed"
    / "all_case_level_metrics.csv"
)
INPUT_GATE = (
    PROJECT_ROOT
    / "experiments"
    / "phase4"
    / "formal_remote_completed"
    / "all_RUN_GATE.json"
)

BOOTSTRAP_SEED = 20260728
PERMUTATION_SEED = 20260729
N_BOOTSTRAP = 2000
N_PERMUTATION = 100000

SYSTEM_ORDER = [
    "B0_ENGINE_ONLY",
    "B1_SINGLE_LLM_NO_TOOLS",
    "B2_SINGLE_LLM_WITH_TOOLS",
    "B3_MULTI_AGENT_NO_VERIFIER",
    "B4_FULL_CLOSED_LOOP",
]
AGENT_ORDER = SYSTEM_ORDER[1:]
ABLATION_ORDER = [
    "B4_FULL_CLOSED_LOOP",
    "B4_NO_EVIDENCE_CONTRACT",
    "B4_NO_PERSISTENT_STRUCTURED_STATE",
    "B4_NO_REVISION_LOOP",
    "B4_NO_VERIFIER",
]

DISPLAY = {
    "B0_ENGINE_ONLY": "B0 Engine only",
    "B1_SINGLE_LLM_NO_TOOLS": "B1 LLM, no tools",
    "B2_SINGLE_LLM_WITH_TOOLS": "B2 Single agent + tools",
    "B3_MULTI_AGENT_NO_VERIFIER": "B3 Multi-agent, no verifier",
    "B4_FULL_CLOSED_LOOP": "B4 Full closed loop",
    "B4_NO_EVIDENCE_CONTRACT": "No evidence contract",
    "B4_NO_PERSISTENT_STRUCTURED_STATE": "No persistent state",
    "B4_NO_REVISION_LOOP": "No revision loop",
    "B4_NO_VERIFIER": "No verifier",
}

BOOLEAN_COLUMNS = [
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
    mapping = {
        True: 1.0,
        False: 0.0,
        "True": 1.0,
        "False": 0.0,
        "true": 1.0,
        "false": 0.0,
        1: 1.0,
        0: 0.0,
    }
    return series.map(mapping)


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    gate = json.loads(INPUT_GATE.read_text(encoding="utf-8"))
    required_gate = {
        "status": "FORMAL_ALL_COMPLETED_UNANALYZED",
        "record_count": 4860,
        "unique_record_count": 4860,
        "api_error_count": 0,
    }
    for key, expected in required_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"Gate mismatch for {key}: {gate.get(key)!r} != {expected!r}")

    data = pd.read_csv(INPUT_CSV)
    if len(data) != 4860:
        raise ValueError(f"Expected 4860 rows, found {len(data)}")
    if data["record_id"].nunique() != 4860:
        raise ValueError("record_id values are not unique")
    if data["api_error_type"].notna().any():
        raise ValueError("API errors are present in the formal data")

    for column in BOOLEAN_COLUMNS:
        if column in data:
            data[column] = normalize_bool(data[column])

    expected_counts = {
        "clean": 1500,
        "ablation": 1200,
        "fault": 2160,
    }
    observed = data.groupby("run_kind").size().to_dict()
    if observed != expected_counts:
        raise ValueError(f"Unexpected run-kind counts: {observed}")
    return data, gate


def cluster_bootstrap_mean(
    data: pd.DataFrame,
    value_col: str,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float, float]:
    case_values = data.groupby("case_id", sort=True)[value_col].mean().dropna()
    values = case_values.to_numpy(dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def cluster_bootstrap_median(
    data: pd.DataFrame,
    value_col: str,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float, float]:
    grouped = {
        key: frame[value_col].dropna().to_numpy(dtype=float)
        for key, frame in data.groupby("case_id", sort=True)
    }
    case_ids = np.array(list(grouped), dtype=object)
    estimate = float(data[value_col].median())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        values = np.concatenate([grouped[case_id] for case_id in sampled])
        draws[idx] = np.median(values)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def paired_case_differences(
    data: pd.DataFrame,
    system_a: str,
    system_b: str,
    value_col: str,
) -> np.ndarray:
    pivot = data.pivot(
        index=["case_id", "formal_repeat"],
        columns="system",
        values=value_col,
    )
    required = [system_a, system_b]
    if not set(required).issubset(pivot.columns):
        raise ValueError(f"Missing systems in paired comparison: {required}")
    pair = pivot[required].dropna()
    expected_pairs = data[data["system"].eq(system_a)][["case_id", "formal_repeat"]]
    if len(pair) != len(expected_pairs):
        raise ValueError(
            f"Pair mismatch for {system_a} versus {system_b}: "
            f"{len(pair)} complete pairs versus {len(expected_pairs)} expected"
        )
    return (
        pair[system_a].sub(pair[system_b]).groupby(level="case_id").mean().to_numpy()
    )


def paired_inference(
    differences: np.ndarray,
    *,
    bootstrap_seed: int,
    permutation_seed: int,
) -> dict[str, float | int]:
    differences = np.asarray(differences, dtype=float)
    estimate = float(differences.mean())
    n_cases = int(differences.size)
    rng = np.random.default_rng(bootstrap_seed)
    boot = rng.choice(
        differences,
        size=(N_BOOTSTRAP, n_cases),
        replace=True,
    ).mean(axis=1)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    rng = np.random.default_rng(permutation_seed)
    observed = abs(estimate)
    exceed = 0
    chunk = 5000
    for start in range(0, N_PERMUTATION, chunk):
        count = min(chunk, N_PERMUTATION - start)
        signs = rng.choice((-1.0, 1.0), size=(count, n_cases))
        permuted = (signs * differences).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
    p_value = float((exceed + 1) / (N_PERMUTATION + 1))
    return {
        "difference": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "n_cases": n_cases,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutation": N_PERMUTATION,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def write_panel_a(clean: pd.DataFrame) -> dict:
    folder = FIGURE_ROOT / "panel_a_primary_success"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, system in enumerate(SYSTEM_ORDER):
        subset = clean[clean["system"].eq(system)]
        estimate, low, high = cluster_bootstrap_mean(
            subset, "verified_task_success", seed=BOOTSTRAP_SEED + idx
        )
        rows.append(
            {
                "system": system,
                "display_name": DISPLAY[system],
                "n_cases": subset["case_id"].nunique(),
                "n_repeats": subset["formal_repeat"].nunique(),
                "n_runs": len(subset),
                "successes": int(subset["verified_task_success"].sum()),
                "rate": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    pd.DataFrame(rows).to_csv(folder / "source_data.csv", index=False)

    differences = paired_case_differences(
        clean,
        "B4_FULL_CLOSED_LOOP",
        "B2_SINGLE_LLM_WITH_TOOLS",
        "verified_task_success",
    )
    comparison = paired_inference(
        differences,
        bootstrap_seed=BOOTSTRAP_SEED + 100,
        permutation_seed=PERMUTATION_SEED + 100,
    )
    pair = clean.pivot(
        index=["case_id", "formal_repeat"],
        columns="system",
        values="verified_task_success",
    )[["B4_FULL_CLOSED_LOOP", "B2_SINGLE_LLM_WITH_TOOLS"]].dropna()
    b4 = pair["B4_FULL_CLOSED_LOOP"].astype(int)
    b2 = pair["B2_SINGLE_LLM_WITH_TOOLS"].astype(int)
    comparison.update(
        {
            "system_a": "B4_FULL_CLOSED_LOOP",
            "system_b": "B2_SINGLE_LLM_WITH_TOOLS",
            "n_paired_runs": len(pair),
            "both_success": int(((b4 == 1) & (b2 == 1)).sum()),
            "b4_only_success": int(((b4 == 1) & (b2 == 0)).sum()),
            "b2_only_success": int(((b4 == 0) & (b2 == 1)).sum()),
            "both_failure": int(((b4 == 0) & (b2 == 0)).sum()),
            "confirmatory": True,
        }
    )
    pd.DataFrame([comparison]).to_csv(folder / "primary_comparison.csv", index=False)
    return comparison


def write_panel_b(clean: pd.DataFrame) -> None:
    folder = FIGURE_ROOT / "panel_b_functional_decomposition"
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {
        "plan_valid": "Plan valid",
        "tool_selection_f1": "Tool F1",
        "schema_valid": "Schema valid",
        "numeric_fidelity": "Numeric fidelity",
        "external_verifier_passed": "Verifier passed",
    }
    rows = []
    for system in AGENT_ORDER:
        subset = clean[clean["system"].eq(system)]
        for key, label in metrics.items():
            rows.append(
                {
                    "system": system,
                    "display_name": DISPLAY[system],
                    "metric": key,
                    "metric_label": label,
                    "n_cases": subset["case_id"].nunique(),
                    "n_runs": subset[key].notna().sum(),
                    "value": subset[key].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(folder / "source_data.csv", index=False)


def write_panel_c(clean: pd.DataFrame) -> None:
    folder = FIGURE_ROOT / "panel_c_grounding_safety"
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {
        "supported_claim_precision": "Supported-claim precision",
        "citation_correctness": "Citation correctness",
        "unsupported_claim_rate": "Unsupported-claim rate",
    }
    rows = []
    seed = BOOTSTRAP_SEED + 300
    for system_index, system in enumerate(AGENT_ORDER):
        subset = clean[clean["system"].eq(system)]
        for metric_index, (key, label) in enumerate(metrics.items()):
            estimate, low, high = cluster_bootstrap_mean(
                subset,
                key,
                seed=seed + system_index * 10 + metric_index,
            )
            rows.append(
                {
                    "system": system,
                    "display_name": DISPLAY[system],
                    "metric": key,
                    "metric_label": label,
                    "n_cases": subset["case_id"].nunique(),
                    "n_runs": subset[key].notna().sum(),
                    "value": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    pd.DataFrame(rows).to_csv(folder / "source_data.csv", index=False)


def write_panel_d(data: pd.DataFrame) -> list[dict]:
    folder = FIGURE_ROOT / "panel_d_ablation"
    folder.mkdir(parents=True, exist_ok=True)
    subset = data[
        (
            data["run_kind"].eq("clean")
            & data["system"].eq("B4_FULL_CLOSED_LOOP")
        )
        | (
            data["run_kind"].eq("ablation")
            & data["system"].isin(ABLATION_ORDER[1:])
        )
    ].copy()
    rows = []
    for idx, system in enumerate(ABLATION_ORDER):
        frame = subset[subset["system"].eq(system)]
        estimate, low, high = cluster_bootstrap_mean(
            frame,
            "verified_task_success",
            seed=BOOTSTRAP_SEED + 400 + idx,
        )
        rows.append(
            {
                "system": system,
                "display_name": DISPLAY[system],
                "n_cases": frame["case_id"].nunique(),
                "n_runs": len(frame),
                "successes": int(frame["verified_task_success"].sum()),
                "rate": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    pd.DataFrame(rows).to_csv(folder / "source_data.csv", index=False)

    comparisons = []
    for idx, ablation in enumerate(ABLATION_ORDER[1:]):
        pair_data = subset[subset["system"].isin(["B4_FULL_CLOSED_LOOP", ablation])]
        differences = paired_case_differences(
            pair_data,
            "B4_FULL_CLOSED_LOOP",
            ablation,
            "verified_task_success",
        )
        result = paired_inference(
            differences,
            bootstrap_seed=BOOTSTRAP_SEED + 500 + idx,
            permutation_seed=PERMUTATION_SEED + 500 + idx,
        )
        result.update(
            {
                "reference": "B4_FULL_CLOSED_LOOP",
                "ablation": ablation,
                "ablation_display_name": DISPLAY[ablation],
                "confirmatory": False,
            }
        )
        comparisons.append(result)
    adjusted = holm_adjust([float(row["p_value"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["p_holm"] = value
    pd.DataFrame(comparisons).to_csv(folder / "paired_comparisons.csv", index=False)
    return comparisons


def write_panel_e(fault: pd.DataFrame) -> None:
    folder = FIGURE_ROOT / "panel_f_fault_matrix"
    folder.mkdir(parents=True, exist_ok=True)
    display_fault = {
        "INVALID_REQUEST_FIELDS": "Invalid request",
        "MISSING_GENE_FEATURES": "Missing genes",
        "PERMANENT_MODEL_FAILURE": "Permanent model failure",
        "MALFORMED_MODEL_OUTPUT": "Malformed model output",
        "TRANSIENT_RETRIEVAL_TIMEOUT": "Retrieval timeout",
        "CITATION_METADATA_MISMATCH": "Citation mismatch",
        "CONFLICTING_EVIDENCE": "Conflicting evidence",
        "UNSUPPORTED_REQUESTED_CLAIM": "Unsupported request",
    }
    rows = []
    grouped = fault.groupby(["fault_type", "system"], sort=False)
    for (fault_type, system), frame in grouped:
        rows.append(
            {
                "fault_type": fault_type,
                "fault_label": display_fault[fault_type],
                "system": system,
                "display_name": DISPLAY[system],
                "n_cases": frame["case_id"].nunique(),
                "n_repeats": frame["formal_repeat"].nunique(),
                "n_runs": len(frame),
                "failure_detection_rate": frame["failure_detected"].mean(),
                "correct_outcome_rate": frame["recovery_or_safe_abstention"].mean(),
                "safe_abstain_rate": frame["safe_abstain"].mean(),
                "median_latency_ms": frame["wall_latency_ms"].median(),
                "mean_total_tokens": frame["total_tokens"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(folder / "source_data.csv", index=False)


def exact_three_run_agreement(frame: pd.DataFrame) -> pd.Series:
    pivot = frame.pivot(
        index="case_id",
        columns="formal_repeat",
        values="verified_task_success",
    )
    if pivot.shape[1] != 3 or pivot.isna().any().any():
        raise ValueError("Expected exactly three complete repeats per case")
    return pivot.nunique(axis=1).eq(1).astype(float)


def write_panel_f(clean: pd.DataFrame) -> None:
    folder = FIGURE_ROOT / "panel_e_reliability_efficiency"
    folder.mkdir(parents=True, exist_ok=True)
    systems = [
        "B2_SINGLE_LLM_WITH_TOOLS",
        "B3_MULTI_AGENT_NO_VERIFIER",
        "B4_FULL_CLOSED_LOOP",
    ]
    rows = []
    for idx, system in enumerate(systems):
        frame = clean[clean["system"].eq(system)]
        agreement = exact_three_run_agreement(frame)
        rng = np.random.default_rng(BOOTSTRAP_SEED + 600 + idx)
        draws = rng.choice(
            agreement.to_numpy(),
            size=(N_BOOTSTRAP, len(agreement)),
            replace=True,
        ).mean(axis=1)
        agreement_low, agreement_high = np.quantile(draws, [0.025, 0.975])
        latency, latency_low, latency_high = cluster_bootstrap_median(
            frame,
            "wall_latency_ms",
            seed=BOOTSTRAP_SEED + 700 + idx,
        )
        rows.append(
            {
                "system": system,
                "display_name": DISPLAY[system],
                "n_cases": frame["case_id"].nunique(),
                "n_runs": len(frame),
                "verified_task_success_rate": frame["verified_task_success"].mean(),
                "exact_three_run_agreement": agreement.mean(),
                "agreement_ci_low": agreement_low,
                "agreement_ci_high": agreement_high,
                "median_latency_ms": latency,
                "latency_ci_low_ms": latency_low,
                "latency_ci_high_ms": latency_high,
                "mean_total_tokens": frame["total_tokens"].mean(),
                "mean_llm_calls": frame["llm_call_count"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(folder / "source_data.csv", index=False)


def main() -> None:
    data, gate = load_and_validate()
    clean = data[data["run_kind"].eq("clean")].copy()
    fault = data[data["run_kind"].eq("fault")].copy()

    primary = write_panel_a(clean)
    write_panel_b(clean)
    write_panel_c(clean)
    ablations = write_panel_d(data)
    write_panel_e(fault)
    write_panel_f(clean)

    summary = {
        "primary_comparison": primary,
        "ablation_comparisons": ablations,
        "interpretation_boundary": (
            "Agent-system benchmark only; no clinical utility, treatment, "
            "deployment, or patient-outcome claim."
        ),
    }
    (FIGURE_ROOT / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "status": "FIGURE2_SOURCE_DATA_READY",
        "input_csv": str(INPUT_CSV),
        "input_csv_sha256": sha256_file(INPUT_CSV),
        "input_gate_sha256": sha256_file(INPUT_GATE),
        "input_gate_status": gate["status"],
        "record_count": len(data),
        "unique_record_count": data["record_id"].nunique(),
        "api_error_count": int(data["api_error_type"].notna().sum()),
        "run_kind_counts": data.groupby("run_kind").size().to_dict(),
        "unique_cases": int(data["case_id"].nunique()),
        "formal_repeats": sorted(data["formal_repeat"].unique().astype(int).tolist()),
        "bootstrap": {
            "unit": "case_id",
            "resamples": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "paired_permutation": {
            "unit": "case_id",
            "draws": N_PERMUTATION,
            "seed": PERMUTATION_SEED,
            "two_sided": True,
        },
        "analysis_script_sha256": sha256_file(Path(__file__)),
    }
    (FIGURE_ROOT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
