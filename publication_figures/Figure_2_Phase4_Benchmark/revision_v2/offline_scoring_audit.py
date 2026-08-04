from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
RAW_JSONL = (
    PROJECT_ROOT
    / "experiments"
    / "phase4"
    / "formal_remote_completed"
    / "all_run_records.jsonl"
)
RUN_GATE = (
    PROJECT_ROOT
    / "experiments"
    / "phase4"
    / "formal_remote_completed"
    / "all_RUN_GATE.json"
)

EXPECTED_TOOLS = ("prognostic_tool", "evidence_tool")
BOOTSTRAP_SEED = 20260728
PERMUTATION_SEED = 20260729
N_BOOTSTRAP = 2000
N_PERMUTATION = 100000

B2 = "B2_SINGLE_LLM_WITH_TOOLS"
B4 = "B4_FULL_CLOSED_LOOP"
ABLATIONS = [
    "B4_NO_EVIDENCE_CONTRACT",
    "B4_NO_PERSISTENT_STRUCTURED_STATE",
    "B4_NO_REVISION_LOOP",
    "B4_NO_VERIFIER",
]
DISPLAY = {
    B2: "B2 Tool-using single controller",
    B4: "B4 Verifier-guided closed loop",
    "B4_NO_EVIDENCE_CONTRACT": "No evidence contract",
    "B4_NO_PERSISTENT_STRUCTURED_STATE": "No persistent state",
    "B4_NO_REVISION_LOOP": "No revision loop",
    "B4_NO_VERIFIER": "No verifier",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def strict_report_contract_complete(state: dict[str, Any]) -> bool:
    """Post-hoc structural audit; does not judge claim truth or task success."""
    report = state.get("final_report")
    if not isinstance(report, dict):
        return False
    if not is_nonempty_string(report.get("report_text")):
        return False
    if not is_nonempty_string(report.get("status")):
        return False

    model = report.get("model_result")
    if not isinstance(model, dict):
        return False
    if model.get("status") != "SUCCESS":
        return False
    if not is_nonempty_string(model.get("case_id")):
        return False
    if not isinstance(model.get("repeat"), int):
        return False
    if not is_nonempty_string(model.get("model_id")):
        return False
    if not is_finite_number(model.get("risk_score")):
        return False
    if not is_sha256(model.get("source_sha256")):
        return False
    if not is_nonempty_string(model.get("provenance")):
        return False

    probabilities = model.get("survival_probabilities")
    if not isinstance(probabilities, dict):
        return False
    if set(probabilities) != {"12m", "36m", "60m"}:
        return False
    if not all(is_finite_number(probabilities[key]) for key in probabilities):
        return False

    claims = report.get("claims")
    if not isinstance(claims, list):
        return False
    for claim in claims:
        if not isinstance(claim, dict):
            return False
        if not is_nonempty_string(claim.get("text")):
            return False
        citation_ids = claim.get("citation_ids")
        if not isinstance(citation_ids, list):
            return False
        if not all(is_nonempty_string(value) for value in citation_ids):
            return False
        if not is_nonempty_string(claim.get("kind")):
            return False
    return True


def classify_plan(
    steps: Any,
    arguments: Any,
    expected: dict[str, Any],
    *,
    plan_status: str | None = None,
) -> str:
    steps = tuple(steps) if isinstance(steps, list) else ()
    arguments = arguments if isinstance(arguments, dict) else {}
    if plan_status and plan_status.startswith("MALFORMED_"):
        return "MALFORMED_PLANNER_JSON"
    if not steps and not arguments:
        return "EMPTY_PLAN_AND_ARGUMENTS"
    if not steps:
        return "EMPTY_TOOL_SEQUENCE"
    if steps != EXPECTED_TOOLS:
        return "WRONG_TOOL_SEQUENCE"
    mismatches = [
        key
        for key in ("case_id", "repeat", "requested_model")
        if arguments.get(key) != expected.get(key)
    ]
    if mismatches:
        return "MISSING_OR_MISMATCHED_ARGUMENTS"
    return "VALID"


def load_records() -> list[dict[str, Any]]:
    gate = json.loads(RUN_GATE.read_text(encoding="utf-8"))
    expected_gate = {
        "status": "FORMAL_ALL_COMPLETED_UNANALYZED",
        "record_count": 4860,
        "unique_record_count": 4860,
        "api_error_count": 0,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"Formal gate mismatch: {key}={gate.get(key)!r}")

    records = []
    with RAW_JSONL.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}") from exc
    if len(records) != 4860:
        raise RuntimeError(f"Expected 4860 records, found {len(records)}")
    record_ids = [record["record_id"] for record in records]
    if len(set(record_ids)) != 4860:
        raise RuntimeError("record_id is not unique")
    if any(record.get("api_error_type") for record in records):
        raise RuntimeError("Formal records contain API errors")
    return records


def derive_audit_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        state = record.get("state") if isinstance(record.get("state"), dict) else {}
        system = str(record.get("system"))
        report_contract = strict_report_contract_complete(state)
        terminal_report = state.get("status") == "VERIFIED_REPORT"
        claim_count = float(record.get("claim_count") or 0)
        exact_support = float(record.get("supported_claim_precision") or 0)
        citation_validity = float(record.get("citation_correctness") or 0)
        citation_completeness = float(record.get("citation_completeness") or 0)
        frozen_pass = bool(record.get("verified_task_success"))
        posthoc_pass = bool(
            system not in {"B0_ENGINE_ONLY", "B1_SINGLE_LLM_NO_TOOLS"}
            and bool(record.get("plan_valid"))
            and bool(record.get("tool_order_exact"))
            and report_contract
            and bool(record.get("numeric_fidelity"))
            and claim_count > 0
            and math.isclose(exact_support, 1.0)
            and math.isclose(citation_validity, 1.0)
            and math.isclose(citation_completeness, 1.0)
            and not bool(record.get("forbidden_claim_case"))
            and terminal_report
        )
        verifier_applicable = system == B4 and record.get("run_kind") == "clean"
        rows.append(
            {
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "case_id": record["case_id"],
                "formal_repeat": int(record["formal_repeat"]),
                "run_kind": record["run_kind"],
                "system": system,
                "fault_type": record.get("fault_type"),
                "frozen_external_composite_pass": frozen_pass,
                "strict_report_contract_complete": report_contract,
                "posthoc_audited_composite_pass": posthoc_pass,
                "internal_verifier_applicable": verifier_applicable,
                "internal_deterministic_verifier_pass": (
                    bool(record.get("external_verifier_passed"))
                    if verifier_applicable
                    else None
                ),
                "exact_extractive_claim_support": exact_support,
                "retrieved_passage_citation_validity": citation_validity,
                "citation_completeness": citation_completeness,
                "unsupported_under_exact_match_contract": float(
                    record.get("unsupported_claim_rate") or 0
                ),
                "plan_valid": bool(record.get("plan_valid")),
                "tool_order_exact": bool(record.get("tool_order_exact")),
                "numeric_fidelity": bool(record.get("numeric_fidelity")),
                "wall_latency_ms": float(record.get("wall_latency_ms") or 0),
                "total_tokens": int(record.get("total_tokens") or 0),
                "failure_detected": (
                    bool(record.get("failure_detected"))
                    if record.get("run_kind") == "fault"
                    else None
                ),
                "correct_terminal_outcome": (
                    bool(record.get("recovery_or_safe_abstention"))
                    if record.get("run_kind") == "fault"
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def plan_audit(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for record in records:
        if record.get("run_kind") != "clean" or record.get("system") not in {B2, B4}:
            continue
        state = record["state"]
        request = state.get("request") or {}
        plan_traces = [
            trace for trace in state.get("traces", []) if trace.get("step") == "plan"
        ]
        first_trace = plan_traces[0] if plan_traces else {}
        first_details = first_trace.get("details") or {}
        initial_class = classify_plan(
            first_details.get("plan"),
            first_details.get("arguments"),
            request,
        )
        final_plan = state.get("plan") or {}
        final_class = classify_plan(
            final_plan.get("steps"),
            final_plan.get("arguments"),
            request,
            plan_status=final_plan.get("status"),
        )
        detail_rows.append(
            {
                "task_id": record["task_id"],
                "case_id": record["case_id"],
                "formal_repeat": int(record["formal_repeat"]),
                "system": record["system"],
                "initial_plan_class": initial_class,
                "planning_revision_count": int(
                    state.get("planning_revision_count") or 0
                ),
                "final_plan_class": final_class,
                "final_plan_valid": bool(record.get("plan_valid")),
                "initial_failure_repaired": (
                    initial_class != "VALID" and bool(record.get("plan_valid"))
                ),
            }
        )
    details = pd.DataFrame(detail_rows)

    summaries = []
    for system, frame in details.groupby("system", sort=False):
        initial_invalid = frame["initial_plan_class"].ne("VALID")
        final_invalid = ~frame["final_plan_valid"]
        repaired = frame["initial_failure_repaired"]
        summaries.append(
            {
                "system": system,
                "display_name": DISPLAY[system],
                "n_runs": len(frame),
                "initial_invalid_n": int(initial_invalid.sum()),
                "initial_invalid_rate": float(initial_invalid.mean()),
                "repaired_n": int(repaired.sum()),
                "repair_rate_among_initial_invalid": (
                    float(repaired.sum() / initial_invalid.sum())
                    if initial_invalid.sum()
                    else 0.0
                ),
                "final_invalid_n": int(final_invalid.sum()),
                "final_invalid_rate": float(final_invalid.mean()),
            }
        )
    return details, pd.DataFrame(summaries)


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    seed: int,
) -> tuple[float, float, float]:
    values = (
        frame.groupby("case_id", sort=True)[value_column]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values, size=(N_BOOTSTRAP, len(values)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def paired_case_differences(
    frame: pd.DataFrame,
    system_a: str,
    system_b: str,
    value_column: str,
) -> np.ndarray:
    pivot = frame.pivot(
        index=["case_id", "formal_repeat"],
        columns="system",
        values=value_column,
    )
    pair = pivot[[system_a, system_b]].dropna()
    return (
        pair[system_a]
        .astype(float)
        .sub(pair[system_b].astype(float))
        .groupby(level="case_id")
        .mean()
        .to_numpy(dtype=float)
    )


def paired_inference(
    differences: np.ndarray,
    *,
    bootstrap_seed: int,
    permutation_seed: int,
) -> dict[str, Any]:
    estimate = float(np.mean(differences))
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(
        differences,
        size=(N_BOOTSTRAP, len(differences)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])

    rng = np.random.default_rng(permutation_seed)
    observed = abs(estimate)
    exceed = 0
    chunk = 5000
    for start in range(0, N_PERMUTATION, chunk):
        count = min(chunk, N_PERMUTATION - start)
        signs = rng.choice((-1.0, 1.0), size=(count, len(differences)))
        permuted = (signs * differences).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
    return {
        "difference": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float((exceed + 1) / (N_PERMUTATION + 1)),
        "n_cases": int(len(differences)),
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutation": N_PERMUTATION,
    }


def exact_three_run_agreement(frame: pd.DataFrame, value_column: str) -> pd.Series:
    pivot = frame.pivot(
        index="case_id",
        columns="formal_repeat",
        values=value_column,
    )
    if pivot.shape[1] != 3 or pivot.isna().any().any():
        raise RuntimeError("Expected three complete formal repeats per case")
    return pivot.nunique(axis=1).eq(1).astype(float)


def write_source_data(audit: pd.DataFrame, records: list[dict[str, Any]]) -> dict:
    clean = audit[audit["run_kind"].eq("clean")].copy()

    # Panel a: architectural contrast, not a quantitative endpoint.
    pd.DataFrame(
        [
            {
                "system": B2,
                "display_name": DISPLAY[B2],
                "controller": "single LLM controller",
                "tools": "frozen prognostic model; frozen evidence assignment",
                "internal_verifier": "not available",
                "conditional_repair": "not available",
                "terminal_states": "direct unverified report or safe abstention",
            },
            {
                "system": B4,
                "display_name": DISPLAY[B4],
                "controller": "role-specialized planner and synthesizer",
                "tools": "frozen prognostic model; frozen evidence assignment",
                "internal_verifier": "deterministic verifier",
                "conditional_repair": "replan; tool retry; one synthesis revision",
                "terminal_states": "verified report or safe abstention",
            },
        ]
    ).to_csv(ROOT / "panel_a_system_contrast" / "source_data.csv", index=False)

    # Panel b: frozen endpoint plus stricter post-hoc structural audit.
    panel_b_rows = []
    for index, system in enumerate([B2, B4]):
        frame = clean[clean["system"].eq(system)]
        for metric, label, status in [
            (
                "frozen_external_composite_pass",
                "Frozen external composite pass",
                "prespecified_confirmatory",
            ),
            (
                "posthoc_audited_composite_pass",
                "Post-hoc strict-contract pass",
                "posthoc_sensitivity",
            ),
        ]:
            estimate, low, high = cluster_bootstrap_mean(
                frame, metric, seed=BOOTSTRAP_SEED + index * 10
            )
            panel_b_rows.append(
                {
                    "system": system,
                    "display_name": DISPLAY[system],
                    "metric": metric,
                    "metric_label": label,
                    "analysis_status": status,
                    "n_cases": frame["case_id"].nunique(),
                    "n_runs": len(frame),
                    "successes": int(frame[metric].sum()),
                    "rate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    pd.DataFrame(panel_b_rows).to_csv(
        ROOT / "panel_b_primary_endpoint" / "source_data.csv", index=False
    )

    differences = paired_case_differences(
        clean[clean["system"].isin([B2, B4])],
        B4,
        B2,
        "frozen_external_composite_pass",
    )
    primary = paired_inference(
        differences,
        bootstrap_seed=BOOTSTRAP_SEED + 100,
        permutation_seed=PERMUTATION_SEED + 100,
    )
    paired = clean[clean["system"].isin([B2, B4])].pivot(
        index=["case_id", "formal_repeat"],
        columns="system",
        values="frozen_external_composite_pass",
    )[[B4, B2]].dropna()
    b4 = paired[B4].astype(int)
    b2 = paired[B2].astype(int)
    primary.update(
        {
            "system_a": B4,
            "system_b": B2,
            "both_pass": int(((b4 == 1) & (b2 == 1)).sum()),
            "b4_only_pass": int(((b4 == 1) & (b2 == 0)).sum()),
            "b2_only_pass": int(((b4 == 0) & (b2 == 1)).sum()),
            "both_fail": int(((b4 == 0) & (b2 == 0)).sum()),
            "n_paired_runs": int(len(paired)),
        }
    )
    pd.DataFrame([primary]).to_csv(
        ROOT / "panel_b_primary_endpoint" / "paired_comparison.csv", index=False
    )

    # Panel c: paired loss relative to full B4.
    combined_records = pd.DataFrame(
        [
            {
                "case_id": record["case_id"],
                "formal_repeat": record["formal_repeat"],
                "run_kind": record["run_kind"],
                "system": record["system"],
                "frozen_external_composite_pass": bool(
                    record.get("verified_task_success")
                ),
            }
            for record in records
            if (
                record.get("run_kind") == "clean" and record.get("system") == B4
            )
            or (
                record.get("run_kind") == "ablation"
                and record.get("system") in ABLATIONS
            )
        ]
    )
    ablation_rows = []
    for index, ablation in enumerate(ABLATIONS):
        pair_frame = combined_records[
            combined_records["system"].isin([B4, ablation])
        ]
        full_minus_ablation = paired_case_differences(
            pair_frame,
            B4,
            ablation,
            "frozen_external_composite_pass",
        )
        result = paired_inference(
            -full_minus_ablation,
            bootstrap_seed=BOOTSTRAP_SEED + 200 + index,
            permutation_seed=PERMUTATION_SEED + 200 + index,
        )
        result.update(
            {
                "ablation": ablation,
                "display_name": DISPLAY[ablation],
                "effect_definition": "ablation minus full B4",
            }
        )
        ablation_rows.append(result)
    p_values = np.asarray([row["p_value"] for row in ablation_rows])
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    for row, p_holm in zip(ablation_rows, adjusted):
        row["p_holm"] = float(p_holm)
    pd.DataFrame(ablation_rows).to_csv(
        ROOT / "panel_c_ablation_effects" / "source_data.csv", index=False
    )

    # Panel d: only metrics with comparable meaning across B2 and B4.
    panel_d_rows = []
    for system_index, system in enumerate([B2, B4]):
        frame = clean[clean["system"].eq(system)]
        metrics = [
            (
                "exact_extractive_claim_support",
                "Exact extractive support",
                "frozen exact-string contract",
            ),
            (
                "retrieved_passage_citation_validity",
                "Retrieved-passage citation validity",
                "citation ID belongs to assigned passages",
            ),
        ]
        for metric_index, (metric, label, reference) in enumerate(metrics):
            estimate, low, high = cluster_bootstrap_mean(
                frame,
                metric,
                seed=BOOTSTRAP_SEED + 300 + system_index * 10 + metric_index,
            )
            panel_d_rows.append(
                {
                    "system": system,
                    "display_name": DISPLAY[system],
                    "metric": metric,
                    "metric_label": label,
                    "reference_standard": reference,
                    "n_cases": frame["case_id"].nunique(),
                    "n_runs": len(frame),
                    "value": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        agreement = exact_three_run_agreement(
            frame, "frozen_external_composite_pass"
        )
        rng = np.random.default_rng(BOOTSTRAP_SEED + 350 + system_index)
        draws = rng.choice(
            agreement.to_numpy(),
            size=(N_BOOTSTRAP, len(agreement)),
            replace=True,
        ).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        panel_d_rows.append(
            {
                "system": system,
                "display_name": DISPLAY[system],
                "metric": "exact_three_run_agreement",
                "metric_label": "Exact three-run agreement",
                "reference_standard": "same binary outcome in all three repeats",
                "n_cases": len(agreement),
                "n_runs": len(frame),
                "value": float(agreement.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    pd.DataFrame(panel_d_rows).to_csv(
        ROOT / "panel_d_traceability_reliability" / "source_data.csv", index=False
    )

    # Panel e: effect of B4 over B2 for each frozen fault.
    fault = audit[
        audit["run_kind"].eq("fault") & audit["system"].isin([B2, B4])
    ].copy()
    fault_labels = {
        "INVALID_REQUEST_FIELDS": "Invalid request",
        "MISSING_GENE_FEATURES": "Missing genes",
        "PERMANENT_MODEL_FAILURE": "Permanent model failure",
        "MALFORMED_MODEL_OUTPUT": "Malformed model output",
        "TRANSIENT_RETRIEVAL_TIMEOUT": "Retrieval timeout",
        "CITATION_METADATA_MISMATCH": "Citation metadata mismatch",
        "CONFLICTING_EVIDENCE": "Conflicting evidence",
        "UNSUPPORTED_REQUESTED_CLAIM": "Unsupported request*",
    }
    fault_order = list(fault_labels)
    panel_e_rows = []
    for fault_index, fault_type in enumerate(fault_order):
        frame = fault[fault["fault_type"].eq(fault_type)]
        for metric_index, (metric, label) in enumerate(
            [
                ("failure_detected", "Failure detection"),
                ("correct_terminal_outcome", "Correct terminal outcome"),
            ]
        ):
            differences = paired_case_differences(frame, B4, B2, metric)
            estimate, low, high = cluster_bootstrap_mean(
                pd.DataFrame(
                    {
                        "case_id": np.arange(len(differences)),
                        "difference": differences,
                    }
                ),
                "difference",
                seed=BOOTSTRAP_SEED + 400 + fault_index * 10 + metric_index,
            )
            panel_e_rows.append(
                {
                    "fault_type": fault_type,
                    "fault_label": fault_labels[fault_type],
                    "metric": metric,
                    "metric_label": label,
                    "difference_b4_minus_b2": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cases": len(differences),
                    "scoring_note": (
                        "B4 detected and safely exited, but the frozen terminal "
                        "outcome rule scored this fault as unsuccessful."
                        if fault_type == "UNSUPPORTED_REQUESTED_CLAIM"
                        else ""
                    ),
                }
            )
    pd.DataFrame(panel_e_rows).to_csv(
        ROOT / "panel_e_fault_handling" / "source_data.csv", index=False
    )
    return primary


def write_audit_report(
    audit: pd.DataFrame,
    plan_details: pd.DataFrame,
    plan_summary: pd.DataFrame,
    primary: dict[str, Any],
) -> None:
    folder = ROOT / "methodological_audit"
    audit.to_csv(folder / "record_level_posthoc_audit.csv", index=False)
    plan_details.to_csv(folder / "planning_error_details.csv", index=False)
    plan_summary.to_csv(folder / "planning_error_summary.csv", index=False)

    clean = audit[audit["run_kind"].eq("clean")]
    metric_rows = [
        {
            "original_name": "schema_valid",
            "revised_name": "strict_report_contract_complete",
            "reference": "post-hoc typed structural contract",
            "status": "post-hoc sensitivity metric",
            "reason": "Original check only required a report object and string text.",
        },
        {
            "original_name": "external_verifier_passed",
            "revised_name": "internal_deterministic_verifier_pass",
            "reference": "B4 internal verifier result",
            "status": "B4-only process diagnostic; N/A for B2/B3",
            "reason": "Absence of a verifier is not a zero-valued quality score.",
        },
        {
            "original_name": "supported_claim_precision",
            "revised_name": "exact_extractive_claim_support",
            "reference": "exact sentence containment in assigned passage",
            "status": "automated extractive contract",
            "reason": "Not an expert biomedical factuality label.",
        },
        {
            "original_name": "citation_correctness",
            "revised_name": "retrieved_passage_citation_validity",
            "reference": "citation ID belongs to the assigned passage set",
            "status": "automated passage-ID validity check",
            "reason": "Does not independently adjudicate biomedical correctness.",
        },
        {
            "original_name": "verified_task_success",
            "revised_name": "frozen_external_composite_pass",
            "reference": "prespecified deterministic composite scorer",
            "status": "confirmatory endpoint retained unchanged",
            "reason": "Avoid confusion with clinical or internal-verifier success.",
        },
    ]
    pd.DataFrame(metric_rows).to_csv(
        folder / "metric_definition_corrections.csv", index=False
    )

    disagreement = (
        clean["frozen_external_composite_pass"]
        != clean["posthoc_audited_composite_pass"]
    )
    plan_text = "\n".join(
        (
            f"- {row.display_name}: {row.initial_invalid_n}/{row.n_runs} initially "
            f"invalid; {row.repaired_n} repaired; {row.final_invalid_n} finally invalid."
        )
        for row in plan_summary.itertuples()
    )
    report = f"""# Phase 4 post-run metric clarification and offline scoring audit

## Status

This is a post-run audit. It does not modify the 4,860 frozen formal records,
their hashes, or the prespecified confirmatory endpoint.

## Planning errors

A planning error is disagreement with the frozen action specification, not
disagreement with a patient outcome or survival label. The required plan is
`prognostic_tool -> evidence_tool` with exact `case_id`, `repeat`, and
`requested_model` arguments.

{plan_text}

B4 repaired {int(plan_summary.loc[plan_summary['system'].eq(B4), 'repaired_n'].iloc[0])}
initially invalid plans through its single allowed replanning step.

## Metric corrections

- `schema_valid` is not used in the revised main figure. The audit instead
  reports `strict_report_contract_complete`, which validates the typed model
  payload, 12/36/60-month probabilities, source hash, provenance, claims,
  citation-ID lists, and non-empty report text.
- `external_verifier_passed` is renamed
  `internal_deterministic_verifier_pass` and is N/A for systems without that
  component.
- `supported_claim_precision` is renamed
  `exact_extractive_claim_support`.
- `citation_correctness` is renamed
  `retrieved_passage_citation_validity`.
- `verified_task_success` is reported as
  `frozen_external_composite_pass`.

## Sensitivity result

The stricter structural audit changed {int(disagreement.sum())} of
{len(clean)} clean-system records. The B4-versus-B2 confirmatory estimate
therefore remains {primary['difference'] * 100:.1f} percentage points
(95% CI {primary['ci_low'] * 100:.1f} to {primary['ci_high'] * 100:.1f});
the strict audit is nevertheless labelled post hoc.

## Interpretation boundary

Evidence metrics assess exact extractive support and passage-ID validity under
the frozen assigned-passage contract. They are not expert-annotated clinical
factuality, semantic retrieval quality, clinical utility, or patient benefit.
"""
    (folder / "OFFLINE_SCORING_AUDIT.md").write_text(report, encoding="utf-8")

    summary = {
        "status": "POSTHOC_OFFLINE_AUDIT_COMPLETED",
        "raw_record_count": 4860,
        "clean_record_count": int(len(clean)),
        "strict_vs_frozen_clean_disagreement_count": int(disagreement.sum()),
        "planning_summary": plan_summary.to_dict(orient="records"),
        "raw_jsonl_sha256": sha256_file(RAW_JSONL),
        "run_gate_sha256": sha256_file(RUN_GATE),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "confirmatory_endpoint_modified": False,
    }
    (folder / "AUDIT_GATE.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    records = load_records()
    audit = derive_audit_table(records)
    plan_details, plan_summary = plan_audit(records)
    primary = write_source_data(audit, records)
    write_audit_report(audit, plan_details, plan_summary, primary)
    print(
        json.dumps(
            {
                "status": "POSTHOC_OFFLINE_AUDIT_COMPLETED",
                "records": len(audit),
                "planning_final_errors": dict(
                    Counter(
                        plan_details.loc[
                            ~plan_details["final_plan_valid"], "system"
                        ]
                    )
                ),
                "strict_vs_frozen_disagreements": int(
                    (
                        audit["frozen_external_composite_pass"]
                        != audit["posthoc_audited_composite_pass"]
                    ).sum()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
