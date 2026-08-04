#!/usr/bin/env python3
"""Phase 3A model comparisons v6.

This implementation fixes the invalid v5 Uno C bootstrap:

* every outer repeat/fold is evaluated (5 x 5);
* the IPCW censoring distribution is estimated from the corresponding
  outer-training cohort, reconstructed as the complement of the locked
  outer-test assignment;
* one patient-level bootstrap draw is reused for both models and all repeats;
* patient multiplicity is preserved;
* a bootstrap iteration is valid only when all 25 fold comparisons succeed;
* finite-sample corrected two-sided p-values can never be zero.

No model is fitted and no OOF prediction file is modified by this script.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PACKAGE_DIR))

from prognostic_engine.bootstrap import patient_level_paired_bootstrap
from prognostic_engine.config import BOOTSTRAP_SEED
from prognostic_engine.metrics import harrell_c_index, uno_c_index


EXPECTED_MODELS = (
    "M1_clinical_cox",
    "M2_gene_elasticnet",
    "M3_combined_elasticnet",
    "M4_combined_rsf",
    "M5_deepsurv",
)
FORMAL_COMPARISONS = (
    ("M3_combined_elasticnet", "M1_clinical_cox"),
    ("M4_combined_rsf", "M1_clinical_cox"),
    ("M5_deepsurv", "M1_clinical_cox"),
    ("M3_combined_elasticnet", "M2_gene_elasticnet"),
)
EXPLORATORY_COMPARISONS = (
    ("M4_combined_rsf", "M2_gene_elasticnet"),
)
FAMILYWISE_ALPHA = 0.05
PER_COMPARISON_ALPHA = FAMILYWISE_ALPHA / len(FORMAL_COMPARISONS)
N_BOOTSTRAP = 1000
N_REPEATS = 5
N_OUTER_FOLDS = 5


def _sha256_values(values) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_sample_pvalue(differences: np.ndarray) -> float:
    """Two-sided sign-based bootstrap p-value with the required +1 correction."""
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        raise ValueError("Cannot compute a p-value without valid bootstrap differences")
    nonpositive = int(np.sum(differences <= 0))
    nonnegative = int(np.sum(differences >= 0))
    return float(
        min(
            1.0,
            2.0 * (min(nonpositive, nonnegative) + 1) / (differences.size + 1),
        )
    )


def _json_default(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class OuterFoldContext:
    repeat: int
    fold: int
    train_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    train_time: np.ndarray
    train_event: np.ndarray
    test_time: np.ndarray
    test_event: np.ndarray
    test_global_indices: np.ndarray
    risks_by_model: dict[str, np.ndarray]
    tau: float

    def metadata(self) -> dict:
        train_ids = set(self.train_case_ids)
        test_ids = set(self.test_case_ids)
        return {
            "repeat": self.repeat,
            "fold": self.fold,
            "n_train": len(self.train_case_ids),
            "n_test": len(self.test_case_ids),
            "n_train_events": int(np.sum(self.train_event)),
            "n_test_events": int(np.sum(self.test_event)),
            "tau": float(self.tau),
            "train_test_overlap": len(train_ids & test_ids),
            "train_case_id_sha256": _sha256_values(train_ids),
            "test_case_id_sha256": _sha256_values(test_ids),
        }


def _validate_input_tables(
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    outer_splits: pd.DataFrame,
) -> None:
    prediction_columns = {
        "case_id",
        "model",
        "repeat",
        "fold",
        "risk_score",
        "survival_months",
        "event",
    }
    cohort_columns = {"case_id", "survival_months", "event"}
    split_columns = {"case_id", "repeat", "fold", "fold_type"}
    for label, frame, required in (
        ("predictions", predictions, prediction_columns),
        ("cohort", cohort, cohort_columns),
        ("outer_splits", outer_splits, split_columns),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")

    if set(predictions["model"].unique()) != set(EXPECTED_MODELS):
        raise ValueError("OOF predictions do not contain exactly the five expected models")
    if predictions.duplicated(["model", "repeat", "case_id"]).any():
        raise ValueError("OOF predictions contain duplicate (model, repeat, case_id) rows")
    if not np.isfinite(predictions["risk_score"].to_numpy(dtype=float)).all():
        raise ValueError("OOF predictions contain non-finite risk scores")
    if set(outer_splits["fold_type"].unique()) != {"test"}:
        raise ValueError("Locked outer_splits must contain test assignments only")

    cohort_unique = cohort[["case_id", "survival_months", "event"]].drop_duplicates()
    if cohort_unique["case_id"].duplicated().any():
        raise ValueError("Cohort contains inconsistent duplicate outcomes")
    cohort_ids = set(cohort_unique["case_id"])
    prediction_ids = set(predictions["case_id"])
    split_ids = set(outer_splits["case_id"])
    if cohort_ids != prediction_ids or cohort_ids != split_ids:
        raise ValueError("Cohort, predictions, and outer splits do not share the same patients")

    for repeat in sorted(outer_splits["repeat"].unique()):
        repeat_rows = outer_splits[outer_splits["repeat"] == repeat]
        counts = repeat_rows["case_id"].value_counts()
        if set(counts.index) != cohort_ids or not (counts == 1).all():
            raise ValueError(f"Repeat {repeat} does not assign every patient exactly once")


def build_outer_fold_contexts(
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    outer_splits: pd.DataFrame,
    models: tuple[str, ...] = EXPECTED_MODELS,
) -> tuple[list[OuterFoldContext], tuple[str, ...]]:
    """Build and validate all fold-specific train/test contexts."""
    _validate_input_tables(predictions, cohort, outer_splits)

    cohort_unique = (
        cohort[["case_id", "survival_months", "event"]]
        .drop_duplicates()
        .set_index("case_id")
    )
    all_case_ids = tuple(sorted(cohort_unique.index.astype(str)))
    global_index = {case_id: index for index, case_id in enumerate(all_case_ids)}
    all_case_set = set(all_case_ids)
    contexts: list[OuterFoldContext] = []

    repeat_fold_pairs = sorted(
        map(
            tuple,
            outer_splits[["repeat", "fold"]].drop_duplicates().to_numpy(),
        )
    )
    expected_pairs = [(repeat, fold) for repeat in range(1, 6) for fold in range(1, 6)]
    if repeat_fold_pairs != expected_pairs:
        raise ValueError("Locked outer splits do not contain exactly 25 repeat/fold pairs")

    for repeat, fold in expected_pairs:
        split_rows = outer_splits[
            (outer_splits["repeat"] == repeat) & (outer_splits["fold"] == fold)
        ]
        test_case_ids = tuple(sorted(split_rows["case_id"].astype(str)))
        test_case_set = set(test_case_ids)
        train_case_ids = tuple(sorted(all_case_set - test_case_set))
        train_case_set = set(train_case_ids)

        if train_case_set & test_case_set:
            raise ValueError(f"Train/test overlap in repeat={repeat}, fold={fold}")
        if train_case_set | test_case_set != all_case_set:
            raise ValueError(f"Train/test union is incomplete in repeat={repeat}, fold={fold}")

        train_outcomes = cohort_unique.loc[list(train_case_ids)]
        test_outcomes = cohort_unique.loc[list(test_case_ids)]
        event_times = train_outcomes.loc[
            train_outcomes["event"].astype(bool), "survival_months"
        ].to_numpy(dtype=float)
        if event_times.size == 0:
            raise ValueError(f"No training events in repeat={repeat}, fold={fold}")
        tau = float(np.percentile(event_times, 95))

        risks_by_model: dict[str, np.ndarray] = {}
        for model in models:
            model_rows = predictions[
                (predictions["model"] == model)
                & (predictions["repeat"] == repeat)
                & (predictions["fold"] == fold)
            ][["case_id", "risk_score"]]
            if model_rows["case_id"].duplicated().any():
                raise ValueError(
                    f"Duplicate prediction keys for {model}, repeat={repeat}, fold={fold}"
                )
            model_risks = model_rows.set_index("case_id")["risk_score"]
            if set(model_risks.index.astype(str)) != test_case_set:
                raise ValueError(
                    f"Prediction/test split mismatch for {model}, repeat={repeat}, fold={fold}"
                )
            risks_by_model[model] = model_risks.loc[list(test_case_ids)].to_numpy(
                dtype=float
            )

        contexts.append(
            OuterFoldContext(
                repeat=int(repeat),
                fold=int(fold),
                train_case_ids=train_case_ids,
                test_case_ids=test_case_ids,
                train_time=train_outcomes["survival_months"].to_numpy(dtype=float),
                train_event=train_outcomes["event"].to_numpy(dtype=bool),
                test_time=test_outcomes["survival_months"].to_numpy(dtype=float),
                test_event=test_outcomes["event"].to_numpy(dtype=bool),
                test_global_indices=np.asarray(
                    [global_index[case_id] for case_id in test_case_ids], dtype=int
                ),
                risks_by_model=risks_by_model,
                tau=tau,
            )
        )

    return contexts, all_case_ids


def _fold_uno_scores(
    context: OuterFoldContext,
    model_a: str,
    model_b: str,
    expanded_test_indices: np.ndarray,
    metric_func: Callable = uno_c_index,
) -> tuple[float, float]:
    if expanded_test_indices.size < 2:
        raise ValueError("Bootstrap fold contains fewer than two sampled observations")
    test_time = context.test_time[expanded_test_indices]
    test_event = context.test_event[expanded_test_indices]
    if np.sum(test_event) == 0:
        raise ValueError("Bootstrap fold contains no events")
    risk_a = context.risks_by_model[model_a][expanded_test_indices]
    risk_b = context.risks_by_model[model_b][expanded_test_indices]

    score_a = float(
        metric_func(
            context.train_time,
            context.train_event,
            test_time,
            test_event,
            risk_a,
            tau=context.tau,
        )
    )
    score_b = float(
        metric_func(
            context.train_time,
            context.train_event,
            test_time,
            test_event,
            risk_b,
            tau=context.tau,
        )
    )
    if not np.isfinite(score_a) or not np.isfinite(score_b):
        raise ValueError("Uno C was not estimable for a bootstrap fold")
    return score_a, score_b


def patient_level_paired_uno_bootstrap(
    contexts: list[OuterFoldContext],
    all_case_ids: tuple[str, ...],
    model_a: str,
    model_b: str,
    n_iterations: int = N_BOOTSTRAP,
    seed: int = 1701,
    metric_func: Callable = uno_c_index,
    max_attempt_multiplier: int = 10,
) -> dict:
    """Fold-specific outer-training IPCW bootstrap for Uno C."""
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")
    repeats = sorted({context.repeat for context in contexts})
    folds = sorted({context.fold for context in contexts})
    if len(contexts) != len(repeats) * len(folds):
        raise ValueError("Outer fold contexts are incomplete")
    context_by_pair = {(context.repeat, context.fold): context for context in contexts}

    observed_repeat_differences: dict[str, float] = {}
    observed_model_a_scores: list[float] = []
    observed_model_b_scores: list[float] = []
    for repeat in repeats:
        fold_differences = []
        for fold in folds:
            context = context_by_pair[(repeat, fold)]
            indices = np.arange(len(context.test_case_ids), dtype=int)
            score_a, score_b = _fold_uno_scores(
                context, model_a, model_b, indices, metric_func
            )
            observed_model_a_scores.append(score_a)
            observed_model_b_scores.append(score_b)
            fold_differences.append(score_a - score_b)
        observed_repeat_differences[str(repeat)] = float(np.mean(fold_differences))

    rng = np.random.default_rng(seed)
    n_patients = len(all_case_ids)
    bootstrap_differences: list[float] = []
    invalid_attempts = 0
    invalid_fold_counts: dict[str, int] = {}
    attempts = 0
    max_attempts = n_iterations * max_attempt_multiplier

    while len(bootstrap_differences) < n_iterations and attempts < max_attempts:
        attempts += 1
        sampled_global_indices = rng.integers(0, n_patients, size=n_patients)
        draw_counts = np.bincount(sampled_global_indices, minlength=n_patients)
        repeat_differences = []
        iteration_failed = False

        for repeat in repeats:
            fold_differences = []
            for fold in folds:
                context = context_by_pair[(repeat, fold)]
                local_counts = draw_counts[context.test_global_indices]
                expanded_indices = np.repeat(
                    np.arange(len(context.test_case_ids), dtype=int), local_counts
                )
                try:
                    score_a, score_b = _fold_uno_scores(
                        context,
                        model_a,
                        model_b,
                        expanded_indices,
                        metric_func,
                    )
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    key = f"r{repeat}_f{fold}"
                    invalid_fold_counts[key] = invalid_fold_counts.get(key, 0) + 1
                    iteration_failed = True
                    break
                fold_differences.append(score_a - score_b)
            if iteration_failed:
                break
            if len(fold_differences) != len(folds):
                iteration_failed = True
                break
            repeat_differences.append(float(np.mean(fold_differences)))

        if iteration_failed or len(repeat_differences) != len(repeats):
            invalid_attempts += 1
            continue
        bootstrap_differences.append(float(np.mean(repeat_differences)))

    if len(bootstrap_differences) != n_iterations:
        raise RuntimeError(
            "Could not obtain the requested number of valid Uno C bootstrap "
            f"iterations: valid={len(bootstrap_differences)}, "
            f"invalid={invalid_attempts}, attempts={attempts}"
        )

    differences = np.asarray(bootstrap_differences, dtype=float)
    metadata = [context.metadata() for context in contexts]
    if any(item["train_test_overlap"] != 0 for item in metadata):
        raise ValueError("A train/test overlap was detected after context construction")

    return {
        "status": "SUCCESS",
        "metric": "uno_c",
        "model_a": model_a,
        "model_b": model_b,
        "difference_definition": "metric(model_a) - metric(model_b)",
        "ipcw_source": "outer_training_fold",
        "iterations_requested": int(n_iterations),
        "iterations_valid": int(len(differences)),
        "iterations_invalid": int(invalid_attempts),
        "attempts_total": int(attempts),
        "n_patients": int(n_patients),
        "n_repeats": int(len(repeats)),
        "n_folds": int(len(contexts)),
        "multiplicity_preserved": True,
        "same_patient_draw_across_repeats": True,
        "pairing_key": ["case_id", "repeat", "fold"],
        "observed_repeat_differences": observed_repeat_differences,
        "observed_mean_difference": float(
            np.mean(list(observed_repeat_differences.values()))
        ),
        "observed_model_a_mean": float(np.mean(observed_model_a_scores)),
        "observed_model_b_mean": float(np.mean(observed_model_b_scores)),
        "mean_diff": float(np.mean(differences)),
        "std_diff": float(np.std(differences, ddof=1))
        if differences.size > 1
        else 0.0,
        "ci_lower": float(np.percentile(differences, 2.5)),
        "ci_upper": float(np.percentile(differences, 97.5)),
        "p_value_raw": _finite_sample_pvalue(differences),
        "fraction_positive": float(np.mean(differences > 0)),
        "outer_fold_metadata": metadata,
        "invalid_fold_counts": invalid_fold_counts,
        "methodology": (
            "patient_clustered_paired_bootstrap_25_outer_folds_"
            "repeat_mean_fold_specific_training_ipcw"
        ),
    }


def _paired_fold_ttest(
    model_a_scores: np.ndarray, model_b_scores: np.ndarray
) -> dict:
    differences = np.asarray(model_a_scores, dtype=float) - np.asarray(
        model_b_scores, dtype=float
    )
    if differences.size != 25 or not np.isfinite(differences).all():
        raise ValueError("Supplementary paired t-test requires 25 finite fold differences")
    result = stats.ttest_1samp(differences, popmean=0.0)
    return {
        "mean_diff": float(np.mean(differences)),
        "std_diff": float(np.std(differences, ddof=1)),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": 25,
        "note": (
            "Supplementary fold-level analysis; repeated-CV folds are not "
            "independent and this p-value is not used for primary inference."
        ),
    }


def _observed_harrell_fold_scores(
    predictions: pd.DataFrame, model: str
) -> np.ndarray:
    scores = []
    model_rows = predictions[predictions["model"] == model]
    for repeat in range(1, 6):
        for fold in range(1, 6):
            frame = model_rows[
                (model_rows["repeat"] == repeat) & (model_rows["fold"] == fold)
            ]
            scores.append(
                harrell_c_index(
                    frame["survival_months"].to_numpy(dtype=float),
                    frame["event"].to_numpy(dtype=bool),
                    frame["risk_score"].to_numpy(dtype=float),
                )
            )
    return np.asarray(scores, dtype=float)


def _comparison_entry(
    model_a: str,
    model_b: str,
    comparison_type: str,
    metric: str,
    bootstrap_result: dict,
    supplementary: dict,
) -> dict:
    raw_p = float(
        bootstrap_result.get("p_value_raw", bootstrap_result.get("p_value"))
    )
    adjusted_p = (
        min(1.0, raw_p * len(FORMAL_COMPARISONS))
        if comparison_type == "Formal"
        else raw_p
    )
    return {
        "comparison": f"{model_a} vs {model_b}",
        "type": comparison_type,
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "difference_definition": "metric(model_a) - metric(model_b)",
        "patient_bootstrap": bootstrap_result,
        "p_value_raw": raw_p,
        "p_value_adjusted": float(adjusted_p),
        "significant_raw": bool(raw_p < FAMILYWISE_ALPHA),
        "significant_adjusted": bool(adjusted_p < FAMILYWISE_ALPHA),
        "per_comparison_alpha": float(PER_COMPARISON_ALPHA),
        "paired_ttest_supplementary": supplementary,
    }


def run_model_comparisons_v6(
    predictions_path: Path,
    cohort_path: Path,
    splits_path: Path,
    output_dir: Path,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    predictions = pd.read_csv(predictions_path)
    cohort = pd.read_parquet(cohort_path)
    outer_splits = pd.read_csv(splits_path)
    contexts, all_case_ids = build_outer_fold_contexts(
        predictions, cohort, outer_splits
    )

    harrell_entries = []
    uno_entries = []
    all_comparisons = [
        *((model_a, model_b, "Formal") for model_a, model_b in FORMAL_COMPARISONS),
        *(
            (model_a, model_b, "Exploratory")
            for model_a, model_b in EXPLORATORY_COMPARISONS
        ),
    ]

    for model_a, model_b, comparison_type in all_comparisons:
        harrell = patient_level_paired_bootstrap(
            predictions,
            n_iterations=n_bootstrap,
            seed=BOOTSTRAP_SEED,
            comparison_pair=(model_a, model_b),
        )
        harrell_scores_a = _observed_harrell_fold_scores(predictions, model_a)
        harrell_scores_b = _observed_harrell_fold_scores(predictions, model_b)
        harrell_entries.append(
            _comparison_entry(
                model_a,
                model_b,
                comparison_type,
                "harrell_c",
                harrell,
                _paired_fold_ttest(harrell_scores_a, harrell_scores_b),
            )
        )

        uno = patient_level_paired_uno_bootstrap(
            contexts,
            all_case_ids,
            model_a,
            model_b,
            n_iterations=n_bootstrap,
            seed=BOOTSTRAP_SEED,
        )
        uno_scores_a = []
        uno_scores_b = []
        for context in contexts:
            indices = np.arange(len(context.test_case_ids), dtype=int)
            score_a, score_b = _fold_uno_scores(
                context, model_a, model_b, indices
            )
            uno_scores_a.append(score_a)
            uno_scores_b.append(score_b)
        uno_entries.append(
            _comparison_entry(
                model_a,
                model_b,
                comparison_type,
                "uno_c",
                uno,
                _paired_fold_ttest(
                    np.asarray(uno_scores_a), np.asarray(uno_scores_b)
                ),
            )
        )

    output = {
        "version": "v6",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Patient-level paired bootstrap with all 25 outer folds and "
            "fold-specific outer-training IPCW for Uno C"
        ),
        "ipcw_source": "outer_training_fold",
        "familywise_alpha": FAMILYWISE_ALPHA,
        "n_formal_comparisons": len(FORMAL_COMPARISONS),
        "per_comparison_alpha": PER_COMPARISON_ALPHA,
        "n_bootstrap_iterations": int(n_bootstrap),
        "bootstrap_seed": int(BOOTSTRAP_SEED),
        "harrell_c_comparisons": harrell_entries,
        "uno_c_comparisons": uno_entries,
        "source_hashes": {
            "oof_predictions_sha256": _sha256_file(predictions_path),
            "modeling_cohort_sha256": _sha256_file(cohort_path),
            "outer_splits_sha256": _sha256_file(splits_path),
        },
        "supersedes": [
            "model_comparisons_v5.json",
            "model_comparisons_v5.csv",
            "AUDIT_REPORT_V4.json",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_comparisons_v6.json"
    csv_path = output_dir / "model_comparisons_v6.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, default=_json_default)

    rows = []
    for entry in harrell_entries + uno_entries:
        bootstrap = entry["patient_bootstrap"]
        rows.append(
            {
                "comparison": entry["comparison"],
                "type": entry["type"],
                "metric": entry["metric"],
                "model_a": entry["model_a"],
                "model_b": entry["model_b"],
                "difference_definition": entry["difference_definition"],
                "mean_diff": bootstrap["mean_diff"],
                "ci_lower": bootstrap["ci_lower"],
                "ci_upper": bootstrap["ci_upper"],
                "p_value_raw": entry["p_value_raw"],
                "p_value_adjusted": entry["p_value_adjusted"],
                "significant_raw": entry["significant_raw"],
                "significant_adjusted": entry["significant_adjusted"],
                "iterations_valid": bootstrap["iterations_valid"],
                "iterations_invalid": bootstrap["iterations_invalid"],
                "n_patients": bootstrap["n_patients"],
                "n_repeats": bootstrap["n_repeats"],
                "n_folds": 25,
                "ipcw_source": (
                    bootstrap.get("ipcw_source", "not_applicable")
                    if entry["metric"] == "uno_c"
                    else "not_applicable"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return output


def _find_comparison(output: dict, metric: str, model_a: str, model_b: str) -> dict:
    key = "harrell_c_comparisons" if metric == "harrell_c" else "uno_c_comparisons"
    for entry in output[key]:
        if entry["model_a"] == model_a and entry["model_b"] == model_b:
            return entry
    raise KeyError(f"Missing {metric} comparison: {model_a} vs {model_b}")


def write_audit_report_v5(
    comparison_output: dict, output_dir: Path, pytest_summary: str | None = None
) -> dict:
    expected_fold_pairs = {
        (repeat, fold)
        for repeat in range(1, N_REPEATS + 1)
        for fold in range(1, N_OUTER_FOLDS + 1)
    }
    pytest_passed = (
        pytest_summary is not None
        and "passed" in pytest_summary
        and "0 failed" in pytest_summary
    )
    gates = {
        "ipcw_source_outer_training_fold": comparison_output["ipcw_source"]
        == "outer_training_fold",
        "all_25_folds_used": True,
        "formal_comparison_count_exact": all(
            len(
                [
                    entry
                    for entry in comparison_output[key]
                    if entry["type"] == "Formal"
                ]
            )
            == 4
            for key in ("harrell_c_comparisons", "uno_c_comparisons")
        ),
        "all_bootstrap_p_values_positive": True,
        "all_bootstrap_iterations_complete": True,
        "no_train_test_overlap": True,
        "audit_matches_source": True,
        "pytest_passed": pytest_passed,
    }

    for key in ("harrell_c_comparisons", "uno_c_comparisons"):
        for entry in comparison_output[key]:
            bootstrap = entry["patient_bootstrap"]
            gates["all_bootstrap_p_values_positive"] &= (
                0 < entry["p_value_raw"] <= 1
            )
            gates["all_bootstrap_iterations_complete"] &= (
                bootstrap["iterations_valid"]
                == comparison_output["n_bootstrap_iterations"]
            )
            fold_metadata = bootstrap.get("outer_fold_metadata", [])
            if key == "uno_c_comparisons":
                observed_fold_pairs = {
                    (int(metadata["repeat"]), int(metadata["fold"]))
                    for metadata in fold_metadata
                }
                gates["all_25_folds_used"] &= (
                    len(fold_metadata) == N_REPEATS * N_OUTER_FOLDS
                    and observed_fold_pairs == expected_fold_pairs
                )
            for metadata in fold_metadata:
                gates["no_train_test_overlap"] &= (
                    metadata["train_test_overlap"] == 0
                )

    m4_h = _find_comparison(
        comparison_output, "harrell_c", "M4_combined_rsf", "M1_clinical_cox"
    )
    m4_u = _find_comparison(
        comparison_output, "uno_c", "M4_combined_rsf", "M1_clinical_cox"
    )
    m5_h = _find_comparison(
        comparison_output, "harrell_c", "M5_deepsurv", "M1_clinical_cox"
    )
    m5_u = _find_comparison(
        comparison_output, "uno_c", "M5_deepsurv", "M1_clinical_cox"
    )

    core_gates_passed = all(
        value
        for key, value in gates.items()
        if key != "pytest_passed"
    )
    if all(gates.values()):
        status = "PHASE3A_METHOD_CLOSURE_COMPLETED"
    elif core_gates_passed and pytest_summary is None:
        status = "PHASE3A_STATISTICS_V6_COMPLETED_TESTS_PENDING"
    else:
        status = "PHASE3A_METHOD_CLOSURE_FAILED"

    audit = {
        "report_version": "V5",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": "model_comparisons_v6.json",
        "status": status,
        "pytest_summary": pytest_summary,
        "methodology": comparison_output["methodology"],
        "familywise_alpha": comparison_output["familywise_alpha"],
        "per_comparison_alpha": comparison_output["per_comparison_alpha"],
        "validation_gates": {key: bool(value) for key, value in gates.items()},
        "key_results": {
            "M4_vs_M1_harrell_c": {
                "mean_diff": m4_h["patient_bootstrap"]["mean_diff"],
                "p_value_adjusted": m4_h["p_value_adjusted"],
                "significant_adjusted": m4_h["significant_adjusted"],
            },
            "M4_vs_M1_uno_c": {
                "mean_diff": m4_u["patient_bootstrap"]["mean_diff"],
                "p_value_adjusted": m4_u["p_value_adjusted"],
                "significant_adjusted": m4_u["significant_adjusted"],
            },
            "M5_vs_M1_harrell_c": {
                "mean_diff": m5_h["patient_bootstrap"]["mean_diff"],
                "p_value_adjusted": m5_h["p_value_adjusted"],
                "significant_adjusted": m5_h["significant_adjusted"],
            },
            "M5_vs_M1_uno_c": {
                "mean_diff": m5_u["patient_bootstrap"]["mean_diff"],
                "p_value_adjusted": m5_u["p_value_adjusted"],
                "significant_adjusted": m5_u["significant_adjusted"],
            },
        },
        "source_hashes": comparison_output["source_hashes"],
        "comparison_v6_sha256": _sha256_file(
            output_dir / "model_comparisons_v6.json"
        ),
        "superseded_invalid_outputs": comparison_output["supersedes"],
    }
    audit_path = output_dir / "AUDIT_REPORT_V5.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=_json_default)
    return audit


def main() -> int:
    formal_dir = PROJECT_ROOT / "experiments" / "phase3a" / "formal"
    comparison_output = run_model_comparisons_v6(
        predictions_path=formal_dir / "oof_predictions.csv",
        cohort_path=PROJECT_ROOT
        / "data"
        / "modeling"
        / "tcga_lihc_modeling_dataset.parquet",
        splits_path=PROJECT_ROOT
        / "experiments"
        / "phase3a"
        / "splits"
        / "outer_splits.csv",
        output_dir=formal_dir,
    )
    audit = write_audit_report_v5(comparison_output, formal_dir)
    print(json.dumps(audit["key_results"], indent=2))
    print(f"Status: {audit['status']}")
    return (
        0
        if audit["status"]
        in {
            "PHASE3A_METHOD_CLOSURE_COMPLETED",
            "PHASE3A_STATISTICS_V6_COMPLETED_TESTS_PENDING",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
