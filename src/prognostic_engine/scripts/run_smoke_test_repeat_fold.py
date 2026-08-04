#!/usr/bin/env python3
"""
Phase 3A Smoke Test - Single Repeat/Fold (Task J)

Executes training with repeat=1, fold=1 to verify the full pipeline works.
Outputs SMOKE_GATE.json to experiments/phase3a/readiness/

Per Phase 3A Reset requirements:
- Uses correct paths: data/modeling and experiments/phase3a/splits
- Calls trainer.load_data() then trainer.run_training_fold(repeat=1, fold=1)
- Validates all 5 models (M1-M5)
- Tracks completed_models, skipped_models, model_failures
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np


def _json_serializer(obj):
    """Serialize objects to JSON, preserving booleans and numpy types."""
    if isinstance(obj, bool):
        return bool(obj)  # Preserve native boolean type
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj) if isinstance(obj, np.integer) else bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(PACKAGE_DIR))

# ACM root is 4 levels up from scripts/ folder
project_root = SCRIPT_DIR.parent.parent.parent.resolve()
ready_dir = project_root / "experiments" / "phase3a" / "readiness"
smoke_gate_path = ready_dir / "SMOKE_GATE.json"

# CORRECT paths per Phase 3A Reset
DATA_PATH = project_root / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
SPLITS_PATH = project_root / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
OUTPUT_DIR = project_root / "experiments" / "phase3a" / "smoke" / "r1_f1"

print(f"Project root: {project_root}")
print(f"Data: {DATA_PATH}")
print(f"Splits: {SPLITS_PATH}")
print(f"Output: {OUTPUT_DIR}")


def run_smoke_test():
    """Run smoke test with repeat=1, fold=1."""
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repeat": 1,
        "fold": 1,
        "tests": {},
        "success": False,
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
    }

    # Test 1: Package imports
    print("\n[1] Testing package imports...")
    try:
        from prognostic_engine import config
        from prognostic_engine import preprocessing
        from prognostic_engine import inner_splits
        from prognostic_engine import models
        from prognostic_engine import metrics
        from prognostic_engine import bootstrap
        from prognostic_engine import training
        results["tests"]["imports"] = {"status": "PASS", "message": "All imports successful"}
        results["passed"] += 1
    except Exception as e:
        results["tests"]["imports"] = {"status": "FAIL", "message": str(e)}
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Test 2: Preprocessing
    print("[2] Testing preprocessing...")
    try:
        from prognostic_engine.preprocessing import preprocess_fold_clinical, preprocess_fold_genes
        from prognostic_engine import config

        np.random.seed(42)
        n = 100
        train_df = pd.DataFrame({
            'case_id': [f'CASE_{i:03d}' for i in range(n)],
            'age_at_diagnosis': np.random.uniform(40, 80, n),
            'ajcc_stage': np.random.choice(['Stage I', 'Stage II', 'Stage IIIA', 'Stage IVA'], n),
            'tumor_grade': np.random.choice(['G1', 'G2', 'G3', 'GX'], n),
            'survival_months': np.random.exponential(24, n),
            'event': np.random.binomial(1, 0.35, n),
            **{f'{g}_log2tpm': np.random.normal(5, 1, n) for g in config.METABOLIC_GENES}
        })
        test_df = pd.DataFrame({
            'case_id': [f'TEST_{i:03d}' for i in range(30)],
            'age_at_diagnosis': np.random.uniform(40, 80, 30),
            'ajcc_stage': np.random.choice(['Stage I', 'Stage II', 'Stage IIIA'], 30),
            'tumor_grade': np.random.choice(['G1', 'G2', 'G3'], 30),
            'survival_months': np.random.exponential(24, 30),
            'event': np.random.binomial(1, 0.35, 30),
            **{f'{g}_log2tpm': np.random.normal(5, 1, 30) for g in config.METABOLIC_GENES}
        })

        clinical_prep = preprocess_fold_clinical(train_df, test_df)
        gene_prep = preprocess_fold_genes(train_df, test_df)

        assert clinical_prep['train_stage'].shape[1] > 1, "Stage should be one-hot"
        assert gene_prep['train_genes'].shape[1] == 15, "Should have 15 gene columns"

        results["tests"]["preprocessing"] = {"status": "PASS", "message": "One-hot encoding working"}
        results["passed"] += 1
    except Exception as e:
        results["tests"]["preprocessing"] = {"status": "FAIL", "message": str(e)}
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Test 3: Inner splits generation
    print("[3] Testing inner splits generation...")
    try:
        from prognostic_engine.inner_splits import generate_inner_splits

        case_ids = [f'CASE_{i:03d}' for i in range(100)]
        inner_splits = generate_inner_splits(case_ids, repeat=1, outer_fold=1)

        assert inner_splits['repeat'] == 1
        assert inner_splits['outer_fold'] == 1
        assert len(inner_splits['folds']) == 5, "Should have 5 inner folds"

        results["tests"]["inner_splits"] = {"status": "PASS", "message": "5 inner folds generated"}
        results["passed"] += 1
    except Exception as e:
        results["tests"]["inner_splits"] = {"status": "FAIL", "message": str(e)}
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Test 4: M2 Coxnet tuning
    print("[4] Testing M2 Coxnet tuning...")
    try:
        from prognostic_engine.models import M2M3Coxnet
        from prognostic_engine.inner_splits import generate_inner_splits

        np.random.seed(42)
        n = 80
        X_train = np.random.randn(n, 15)
        y_train = np.random.exponential(24, n)
        event_train = np.random.binomial(1, 0.35, n)

        # Generate inner CV splits
        case_ids = [f'CASE_{i:03d}' for i in range(n)]
        inner_cv_splits = generate_inner_splits(case_ids, repeat=1, outer_fold=1)

        # Use reduced grid for speed
        original_alpha = config.M2_M3_ALPHA_RANGE
        original_l1 = config.M2_M3_L1_RATIO_RANGE
        config.M2_M3_ALPHA_RANGE = [0.01, 0.1]
        config.M2_M3_L1_RATIO_RANGE = [0.2, 0.5]

        model = M2M3Coxnet('M2')
        best_alpha, best_l1 = model.tune(X_train, y_train, event_train, inner_cv_splits=inner_cv_splits)

        # Restore
        config.M2_M3_ALPHA_RANGE = original_alpha
        config.M2_M3_L1_RATIO_RANGE = original_l1

        assert best_alpha is not None
        assert best_l1 is not None

        results["tests"]["m2_tuning"] = {"status": "PASS", "message": f"alpha={best_alpha}, l1={best_l1:.2f}"}
        results["passed"] += 1
    except Exception as e:
        config.M2_M3_ALPHA_RANGE = original_alpha
        config.M2_M3_L1_RATIO_RANGE = original_l1
        results["tests"]["m2_tuning"] = {"status": "FAIL", "message": str(e)}
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Test 5: Metrics computation
    print("[5] Testing metrics computation...")
    try:
        from prognostic_engine.metrics import harrell_c_index, uno_c_index, time_dependent_auc, compute_brier_score

        np.random.seed(42)
        n = 100
        y_time = np.random.exponential(24, n)
        y_event = np.random.binomial(1, 0.35, n)
        risk_scores = np.random.randn(n)

        harrell_c = harrell_c_index(y_time, y_event, risk_scores)
        assert 0 <= harrell_c <= 1

        # Test Uno C with IPCW (requires separate train/test for proper estimation)
        # For smoke test, we use same data for both (acceptable for smoke testing)
        uno_c = uno_c_index(y_time, y_event, y_time, y_event, risk_scores)
        assert 0 <= uno_c <= 1

        # Test time-dependent AUC
        auc_results = time_dependent_auc(y_time, y_event, y_time, y_event, risk_scores, times=[12, 36, 60])
        assert 'auc_12m' in auc_results or 'auc_12m_status' in auc_results

        # Test Brier score (survival probs shape should match test samples)
        survival_probs = np.random.uniform(0.2, 0.9, (n, 3))  # n samples x 3 timepoints
        brier_results = compute_brier_score(y_time, y_event, y_time, y_event, survival_probs, times=[12, 36, 60])
        assert 'brier_12m' in brier_results or 'ibs' in brier_results

        results["tests"]["metrics"] = {"status": "PASS", "message": f"C={harrell_c:.3f}, Uno_C={uno_c:.3f}"}
        results["passed"] += 1
    except Exception as e:
        results["tests"]["metrics"] = {"status": "FAIL", "message": str(e)}
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Test 6: Bootstrap with repeat/fold
    print("[6] Testing bootstrap with repeat/fold support...")
    try:
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        np.random.seed(42)
        cases = [f'CASE_{i:03d}' for i in range(50)]
        predictions = []
        outcomes = {
            case_id: (float(np.random.exponential(24)), int(np.random.binomial(1, 0.35)))
            for case_id in cases
        }
        # Each patient appears in exactly one OOF fold per repeat, and paired
        # models share the same observed outcome.
        for repeat in range(1, 3):
            for case_index, case_id in enumerate(cases):
                fold = case_index % 5 + 1
                survival_months, event = outcomes[case_id]
                for model_name in ['M1', 'M2']:
                        predictions.append({
                            'case_id': case_id,
                            'model': model_name,
                            'risk_score': np.random.randn(),
                            'survival_months': survival_months,
                            'event': event,
                            'repeat': repeat,
                            'fold': fold
                        })
        pred_df = pd.DataFrame(predictions)

        # Test with repeat/fold metadata
        result = patient_level_paired_bootstrap(pred_df, n_iterations=100, seed=456, repeat=1, fold=1)

        assert 'repeat' in result, "Missing repeat metadata"
        assert 'fold' in result, "Missing fold metadata"
        assert result['repeat'] == 1
        assert result['fold'] == 1

        results["tests"]["bootstrap"] = {"status": "PASS", "message": f"repeat={result['repeat']}, fold={result['fold']}"}
        results["passed"] += 1
    except Exception as e:
        results["tests"]["bootstrap"] = {"status": "FAIL", "message": str(e)}
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Test 7: IntegrityMonitor non-blocking behavior
    print("[7] Testing IntegrityMonitor non-blocking behavior...")
    try:
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M1', repeat=1, fold=1)
        monitor.check_c_index(0.3)  # Very poor
        monitor.check_auc({'auc_12': 0.4, 'auc_36': 0.4})  # Very poor
        monitor.check_ibs(0.9)  # Very poor

        status = monitor.get_status()
        assert status['status'] == 'MONITORED', f"Should be MONITORED, got {status['status']}"

        results["tests"]["integrity_monitor"] = {"status": "PASS", "message": "Never blocks"}
        results["passed"] += 1
    except Exception as e:
        results["tests"]["integrity_monitor"] = {"status": "FAIL", "message": str(e)}
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Test 8: NestedCVTrainer with repeat=1, fold=1 (actual training)
    print("[8] Testing NestedCVTrainer (repeat=1, fold=1)...")
    try:
        from prognostic_engine.training import NestedCVTrainer

        # Per Phase 3A Reset: Use correct paths
        data_path = str(DATA_PATH)
        splits_path = str(SPLITS_PATH)

        # Check paths exist first
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        if not SPLITS_PATH.exists():
            raise FileNotFoundError(f"Splits file not found: {SPLITS_PATH}")

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Instantiate trainer
        trainer = NestedCVTrainer(
            data_path=data_path,
            splits_path=splits_path,
            output_dir=str(OUTPUT_DIR)
        )

        # Load data
        print("  Loading data...")
        trainer.load_data()

        # Verify data loaded
        n_patients = len(trainer.df)
        n_events = trainer.df['event'].sum()
        print(f"  Loaded: {n_patients} patients, {n_events} events")

        # Run single fold
        print("  Running training fold (repeat=1, fold=1)...")
        fold_results = trainer.run_training_fold(repeat=1, fold=1)

        # Get test set size
        test_cases = trainer.splits[
            (trainer.splits['repeat'] == 1) &
            (trainer.splits['fold'] == 1) &
            (trainer.splits['fold_type'] == 'test')
        ]['case_id'].values
        n_test = len(test_cases)
        print(f"  Test patients: {n_test}")

        # Use the same canonical persistence/round-trip path as formal training.
        # Exact pilot/formal aggregation gates are covered by integration tests,
        # because a one-fold smoke run is intentionally incomplete.
        print("  Running canonical prediction persistence...")
        pred_df, pred_path, roundtrip_ok = trainer._write_predictions()
        if not roundtrip_ok:
            raise RuntimeError("Prediction CSV round-trip validation failed")

        n_predictions = len(pred_df)
        print(f"  Predictions saved: {n_predictions} to {pred_path}")

        # Per-model prediction counts
        per_model_counts = pred_df.groupby('model').size().to_dict()
        print(f"  Per-model counts: {per_model_counts}")

        # Check for NaN/Inf in predictions
        nan_risk_scores = pred_df['risk_score'].isna().sum()
        inf_risk_scores = np.isinf(pred_df['risk_score']).sum() if nan_risk_scores == 0 else 0

        # Check for duplicate case_id per model (each patient should appear once per model)
        duplicates = {}
        for model_name, group in pred_df.groupby('model'):
            dup_count = group['case_id'].duplicated().sum()
            if dup_count > 0:
                duplicates[model_name] = dup_count

        # Strict validation: Determine model completion status
        expected_models = ['M1_clinical_cox', 'M2_gene_elasticnet', 'M3_combined_elasticnet',
                          'M4_combined_rsf', 'M5_deepsurv']
        completed_models = []
        skipped_models = []
        model_failures = {}

        for model_key in expected_models:
            if model_key in trainer.metrics_by_model and trainer.metrics_by_model[model_key]:
                # Check if we have valid metrics (not NaN/Inf)
                metrics = trainer.metrics_by_model[model_key][-1]
                if metrics and not np.isnan(metrics.get('harrell_c', np.nan)):
                    completed_models.append(model_key)
                else:
                    model_failures[model_key] = trainer.model_failures.get(model_key, [])
            else:
                skipped_models.append(model_key)

        # Count NaN metrics per model
        nan_metrics = {}
        primary_metric_keys = (
            'harrell_c', 'uno_c', 'auc_12m', 'auc_36m', 'auc_60m',
            'brier_12m', 'brier_36m', 'brier_60m', 'ibs',
        )
        for model_key in expected_models:
            if model_key in trainer.metrics_by_model and trainer.metrics_by_model[model_key]:
                nan_count = 0
                for metric_record in trainer.metrics_by_model[model_key]:
                    if metric_record is None:
                        nan_count += len(primary_metric_keys)
                        continue
                    nan_count += sum(
                        1 for key in primary_metric_keys
                        if metric_record.get(key) is None
                        or not np.isfinite(float(metric_record[key]))
                    )
                if nan_count > 0:
                    nan_metrics[model_key] = nan_count

        print(f"  Completed models: {completed_models}")
        print(f"  Skipped models: {skipped_models}")
        print(f"  Failed models: {list(model_failures.keys())}")
        print(f"  NaN metrics: {nan_metrics}")

        # STRICT SUCCESS CRITERIA (per user requirements):
        # 1. ALL 5 models must complete (no skip, no fail)
        # 2. Per-model predictions must equal n_test (73)
        # 3. Total predictions must equal 5 * n_test (365)
        # 4. No NaN/Inf in predictions or metrics
        # 5. No duplicate case_id per model
        # 6. model_failures must be empty
        # 7. skipped_models must be empty

        expected_total = 5 * n_test
        per_model_mismatch = {m: per_model_counts.get(m, 0) for m in expected_models
                              if per_model_counts.get(m, 0) != n_test}

        validation_results = {
            "all_models_completed": bool(len(completed_models) == 5 and len(skipped_models) == 0 and len(model_failures) == 0),
            "per_model_predictions_equal": bool(len(per_model_mismatch) == 0),
            "total_predictions_equal": bool(n_predictions == expected_total),
            "no_nan_in_predictions": bool(nan_risk_scores == 0 and inf_risk_scores == 0),
            "no_nan_in_metrics": bool(len(nan_metrics) == 0),
            "no_duplicate_records": bool(len(duplicates) == 0),
            "model_failures_empty": bool(len(model_failures) == 0),
            "skipped_models_empty": bool(len(skipped_models) == 0),
        }

        all_valid = all(validation_results.values())

        print(f"\n  STRICT VALIDATION RESULTS:")
        for check, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {check}: {status}")

        # Build detailed test result
        test_detail = {
            "status": "PASS" if all_valid else "FAIL",
            "n_test_patients": n_test,
            "expected_models": expected_models,
            "completed_models": completed_models,
            "skipped_models": skipped_models,
            "model_failures": {k: [{"error": str(f["error"])[:200]} for f in v] for k, v in model_failures.items()},
            "per_model_counts": per_model_counts,
            "per_model_expected": n_test,
            "total_predictions": n_predictions,
            "expected_total": expected_total,
            "nan_in_predictions": {"risk_score_nan": int(nan_risk_scores), "risk_score_inf": int(inf_risk_scores)},
            "nan_in_metrics": nan_metrics,
            "duplicate_records": duplicates,
            "validation": validation_results,
        }

        if all_valid:
            results["tests"]["nested_cv_trainer"] = {
                "status": "PASS",
                "message": f"ALL 5 models completed, {n_predictions} predictions (73 each), no NaN/Inf, no duplicates",
                **test_detail
            }
        else:
            results["tests"]["nested_cv_trainer"] = {
                "status": "FAIL",
                "message": f"Validation failed: {[k for k, v in validation_results.items() if not v]}",
                **test_detail
            }
            results["failed"] += 1
            return results

        results["passed"] += 1
    except Exception as e:
        import traceback
        results["tests"]["nested_cv_trainer"] = {
            "status": "FAIL",
            "message": f"{str(e)}\n{traceback.format_exc()}"
        }
        results["failed"] += 1
        return results
    results["total_tests"] += 1

    # Add additional smoke test metadata (from test 8)
    # Variables are available because test 8 passed (we're past the return statement)
    results["n_test_patients"] = n_test
    results["expected_models"] = expected_models
    results["completed_models"] = completed_models
    results["skipped_models"] = skipped_models
    results["model_failures"] = {k: [{"error": str(f["error"])[:200]} for f in v] for k, v in model_failures.items()}
    results["total_predictions"] = n_predictions
    results["expected_total"] = expected_total
    results["validation"] = validation_results

    results["success"] = results["passed"] == results["total_tests"]
    return results


def main():
    """Run smoke test and write gate."""
    print("=" * 70)
    print("PHASE 3A SMOKE TEST - REPEAT=1, FOLD=1")
    print("=" * 70)

    ready_dir.mkdir(parents=True, exist_ok=True)

    results = run_smoke_test()

    # Write SMOKE_GATE.json
    with open(smoke_gate_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_serializer)

    # Print summary
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    for test_name, result in results["tests"].items():
        status = result["status"]
        msg = result["message"]
        print(f"  {test_name:<25} [{status}] {msg}")

    print(f"\nTotal: {results['passed']}/{results['total_tests']} passed")

    if results["success"]:
        print("\n>>> SMOKE TEST PASSED <<<")
        print(f"Gate written to: {smoke_gate_path}")
        return 0
    else:
        print("\n>>> SMOKE TEST FAILED <<<")
        print(f"Gate written to: {smoke_gate_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
