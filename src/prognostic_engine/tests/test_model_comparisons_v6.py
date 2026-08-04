"""Methodological contract tests for Phase 3A comparison v6."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ACM_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ACM_ROOT / "src" / "prognostic_engine" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_model_comparisons_v6 as v6


@pytest.fixture(scope="module")
def real_inputs():
    formal = ACM_ROOT / "experiments" / "phase3a" / "formal"
    predictions = pd.read_csv(formal / "oof_predictions.csv")
    cohort = pd.read_parquet(
        ACM_ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    )
    splits = pd.read_csv(
        ACM_ROOT / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
    )
    return predictions, cohort, splits


@pytest.fixture(scope="module")
def real_contexts(real_inputs):
    return v6.build_outer_fold_contexts(*real_inputs)


def test_contexts_cover_all_25_outer_folds(real_contexts):
    contexts, case_ids = real_contexts
    assert len(contexts) == 25
    assert len(case_ids) == 363
    assert {(context.repeat, context.fold) for context in contexts} == {
        (repeat, fold) for repeat in range(1, 6) for fold in range(1, 6)
    }


def test_every_context_uses_disjoint_outer_training_complement(real_contexts):
    contexts, case_ids = real_contexts
    complete = set(case_ids)
    for context in contexts:
        train_ids = set(context.train_case_ids)
        test_ids = set(context.test_case_ids)
        assert not train_ids & test_ids
        assert train_ids | test_ids == complete
        assert len(train_ids) in (290, 291)
        assert len(test_ids) in (72, 73)
        assert context.metadata()["train_test_overlap"] == 0


def test_tau_is_derived_from_each_outer_training_fold(real_contexts):
    contexts, _ = real_contexts
    for context in contexts:
        expected = np.percentile(
            context.train_time[context.train_event.astype(bool)], 95
        )
        assert context.tau == pytest.approx(expected)
    # A single test-derived global tau would make all values identical.
    assert len({round(context.tau, 8) for context in contexts}) > 5


def test_bootstrap_calls_metric_for_all_folds_and_outer_train_only(real_contexts):
    contexts, case_ids = real_contexts
    calls = []

    def metric_spy(
        train_time,
        train_event,
        test_time,
        test_event,
        risk_scores,
        tau=None,
    ):
        calls.append(
            {
                "n_train": len(train_time),
                "n_test": len(test_time),
                "tau": tau,
            }
        )
        # A deterministic finite score suitable for testing orchestration.
        return float(np.mean(risk_scores))

    result = v6.patient_level_paired_uno_bootstrap(
        contexts,
        case_ids,
        "M4_combined_rsf",
        "M1_clinical_cox",
        n_iterations=1,
        seed=91,
        metric_func=metric_spy,
    )

    # 25 observed folds + 25 bootstrap folds, with two model calls each.
    assert len(calls) == 100
    assert {call["n_train"] for call in calls} == {290, 291}
    assert all(call["n_test"] > 0 for call in calls)
    assert len({round(call["tau"], 8) for call in calls}) > 5
    assert result["n_folds"] == 25
    assert result["n_repeats"] == 5


def test_real_uno_bootstrap_has_positive_finite_sample_pvalue(real_contexts):
    contexts, case_ids = real_contexts
    result = v6.patient_level_paired_uno_bootstrap(
        contexts,
        case_ids,
        "M5_deepsurv",
        "M1_clinical_cox",
        n_iterations=3,
        seed=92,
    )
    assert result["iterations_valid"] == 3
    assert result["iterations_invalid"] >= 0
    assert result["p_value_raw"] > 0
    assert result["p_value_raw"] >= 2 / 4
    assert len(result["outer_fold_metadata"]) == 25
    assert result["ipcw_source"] == "outer_training_fold"


def test_finite_sample_pvalue_can_never_be_zero():
    always_negative = np.full(1000, -1.0)
    assert v6._finite_sample_pvalue(always_negative) == pytest.approx(2 / 1001)


def test_source_has_no_fold_one_or_default_weight_fallback():
    source = inspect.getsource(v6.patient_level_paired_uno_bootstrap)
    assert "fold = 1" not in source
    assert "weight = 1.0" not in source
    assert "get((repeat, fold, cid), 1.0)" not in source
    assert "for fold in folds" in source


def test_v6_uses_locked_bootstrap_seed():
    source = inspect.getsource(v6.run_model_comparisons_v6)
    assert "seed=BOOTSTRAP_SEED" in source
    assert v6.BOOTSTRAP_SEED == 456


def test_duplicate_or_incomplete_outer_assignment_is_rejected(real_inputs):
    predictions, cohort, splits = real_inputs
    broken = splits.iloc[1:].copy()
    with pytest.raises(ValueError, match="assign every patient exactly once"):
        v6.build_outer_fold_contexts(predictions, cohort, broken)


def _minimal_comparison_output(tmp_path: Path, raw_p: float) -> dict:
    def entry(metric, model_a, model_b):
        bootstrap = {
            "status": "SUCCESS",
            "metric": metric,
            "model_a": model_a,
            "model_b": model_b,
            "iterations_requested": 2,
            "iterations_valid": 2,
            "iterations_invalid": 0,
            "n_patients": 363,
            "n_repeats": 5,
            "n_folds": 25,
            "mean_diff": 0.01,
            "p_value_raw": raw_p,
            "outer_fold_metadata": [
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_test_overlap": 0,
                }
                for repeat in range(1, 6)
                for fold in range(1, 6)
            ],
        }
        return {
            "comparison": f"{model_a} vs {model_b}",
            "type": "Formal",
            "metric": metric,
            "model_a": model_a,
            "model_b": model_b,
            "patient_bootstrap": bootstrap,
            "p_value_raw": raw_p,
            "p_value_adjusted": min(1.0, raw_p * 4),
            "significant_adjusted": False,
        }

    comparisons = [
        ("M3_combined_elasticnet", "M1_clinical_cox"),
        ("M4_combined_rsf", "M1_clinical_cox"),
        ("M5_deepsurv", "M1_clinical_cox"),
        ("M3_combined_elasticnet", "M2_gene_elasticnet"),
    ]
    output = {
        "methodology": "test",
        "ipcw_source": "outer_training_fold",
        "familywise_alpha": 0.05,
        "per_comparison_alpha": 0.0125,
        "n_bootstrap_iterations": 2,
        "harrell_c_comparisons": [
            entry("harrell_c", model_a, model_b)
            for model_a, model_b in comparisons
        ],
        "uno_c_comparisons": [
            entry("uno_c", model_a, model_b) for model_a, model_b in comparisons
        ],
        "source_hashes": {},
        "supersedes": [],
    }
    (tmp_path / "model_comparisons_v6.json").write_text(
        json.dumps(output), encoding="utf-8"
    )
    return output


def test_audit_fails_when_any_bootstrap_pvalue_is_zero(tmp_path):
    output = _minimal_comparison_output(tmp_path, raw_p=0.0)
    audit = v6.write_audit_report_v5(output, tmp_path)
    assert audit["status"] == "PHASE3A_METHOD_CLOSURE_FAILED"
    assert audit["validation_gates"]["all_bootstrap_p_values_positive"] is False


def test_audit_is_generated_from_source_values(tmp_path):
    output = _minimal_comparison_output(tmp_path, raw_p=0.2)
    audit = v6.write_audit_report_v5(output, tmp_path)
    source_entry = output["uno_c_comparisons"][1]
    assert (
        audit["key_results"]["M4_vs_M1_uno_c"]["p_value_adjusted"]
        == source_entry["p_value_adjusted"]
    )
