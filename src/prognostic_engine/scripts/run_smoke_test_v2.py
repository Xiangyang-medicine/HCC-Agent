#!/usr/bin/env python3
"""
Phase 3A Smoke Test v2 - Formal Nested CV Components

Tests all components of the formal nested-CV implementation:
1. Package imports
2. Preprocessing (one-hot encoding)
3. Inner CV splits generation
4. Model tuning (M2 Coxnet)
5. Metrics computation (Uno C, AUC, Brier)
6. Bootstrap comparison

This must PASS before running full 5×5×5 nested CV.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add package to path - use src directory
PACKAGE_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(PACKAGE_DIR))

def test_imports():
    """Test all package imports."""
    print("\n[1/6] Testing package imports...")

    try:
        from prognostic_engine import config
        from prognostic_engine import preprocessing
        from prognostic_engine import inner_splits
        from prognostic_engine import models
        from prognostic_engine import metrics
        from prognostic_engine import bootstrap
        from prognostic_engine import training

        assert hasattr(config, 'METABOLIC_GENES')
        assert hasattr(config, 'N_OUTER_REPEATS')
        assert len(config.METABOLIC_GENES) == 15

        print("  PASS: All imports successful")
        return True
    except Exception as e:
        print(f"  FAIL: Import error - {e}")
        return False


def test_preprocessing():
    """Test one-hot encoding preprocessing."""
    print("\n[2/6] Testing preprocessing (one-hot encoding)...")

    try:
        from prognostic_engine.preprocessing import (
            preprocess_fold_clinical, preprocess_fold_genes
        )

        # Create mock data
        from prognostic_engine.config import METABOLIC_GENES
        np.random.seed(42)
        n = 100

        train_df = pd.DataFrame({
            'case_id': [f'CASE_{i:03d}' for i in range(n)],
            'age_at_diagnosis': np.random.uniform(40, 80, n),
            'ajcc_stage': np.random.choice(['Stage I', 'Stage II', 'Stage IIIA', 'Stage IVA'], n),
            'tumor_grade': np.random.choice(['G1', 'G2', 'G3', 'GX'], n),
            'survival_months': np.random.exponential(24, n),
            'event': np.random.binomial(1, 0.35, n),
            **{f'{g}_log2tpm': np.random.normal(5, 1, n) for g in METABOLIC_GENES}
        })

        test_df = pd.DataFrame({
            'case_id': [f'TEST_{i:03d}' for i in range(30)],
            'age_at_diagnosis': np.random.uniform(40, 80, 30),
            'ajcc_stage': np.random.choice(['Stage I', 'Stage II', 'Stage IIIA'], 30),
            'tumor_grade': np.random.choice(['G1', 'G2', 'G3'], 30),
            'survival_months': np.random.exponential(24, 30),
            'event': np.random.binomial(1, 0.35, 30),
            **{f'{g}_log2tpm': np.random.normal(5, 1, 30) for g in METABOLIC_GENES}
        })

        # Test clinical preprocessing
        clinical_prep = preprocess_fold_clinical(train_df, test_df)

        # Verify one-hot encoding
        assert 'stage_cols' in clinical_prep, "Stage columns missing"
        assert 'grade_cols' in clinical_prep, "Grade columns missing"
        assert clinical_prep['train_stage'].shape[1] > 1, "Stage should be one-hot (multiple columns)"
        assert clinical_prep['train_grade'].shape[1] > 1, "Grade should be one-hot (multiple columns)"

        # Test gene preprocessing
        gene_prep = preprocess_fold_genes(train_df, test_df)
        assert gene_prep['train_genes'].shape[1] == 15, "Should have 15 gene columns"

        print("  PASS: One-hot encoding working correctly")
        print(f"    Stage columns: {clinical_prep['stage_cols']}")
        print(f"    Grade columns: {clinical_prep['grade_cols']}")
        return True

    except Exception as e:
        print(f"  FAIL: Preprocessing error - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inner_splits():
    """Test inner CV split generation."""
    print("\n[3/6] Testing inner CV split generation...")

    try:
        from prognostic_engine.inner_splits import generate_inner_splits

        case_ids = [f'CASE_{i:03d}' for i in range(100)]

        inner_splits = generate_inner_splits(case_ids, repeat=1, outer_fold=1)

        assert inner_splits['repeat'] == 1
        assert inner_splits['outer_fold'] == 1
        assert len(inner_splits['folds']) == 5, "Should have 5 inner folds"

        # Verify all cases are covered
        total_train = sum(len(f['train_indices']) for f in inner_splits['folds'])
        total_val = sum(len(f['val_indices']) for f in inner_splits['folds'])

        assert total_train + total_val == len(case_ids) * 5, "All cases should appear in each fold"

        print("  PASS: Inner CV splits generated correctly")
        print(f"    Cases: {len(case_ids)}, Inner folds: {len(inner_splits['folds'])}")
        return True

    except Exception as e:
        print(f"  FAIL: Inner splits error - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_m2_tuning():
    """Test M2 Coxnet inner CV tuning."""
    print("\n[4/6] Testing M2 Coxnet inner CV tuning...")

    try:
        from prognostic_engine.models import M2M3Coxnet
        from prognostic_engine.config import M2_M3_ALPHA_RANGE, M2_M3_L1_RATIO_RANGE

        # Create mock data (simpler for speed)
        np.random.seed(42)
        n = 80

        X_train = np.random.randn(n, 15)
        y_train = np.random.exponential(24, n)
        event_train = np.random.binomial(1, 0.35, n)

        X_test = np.random.randn(20, 15)
        y_test = np.random.exponential(24, 20)
        event_test = np.random.binomial(1, 0.35, 20)

        model = M2M3Coxnet('M2')

        # Test tuning (use reduced grid for speed)
        original_alpha = M2_M3_ALPHA_RANGE
        original_l1 = M2_M3_L1_RATIO_RANGE

        # Temporarily reduce for speed
        import prognostic_engine.config as cfg
        cfg.M2_M3_ALPHA_RANGE = [0.01, 0.1]
        cfg.M2_M3_L1_RATIO_RANGE = [0.2, 0.5]

        best_alpha, best_l1 = model.tune(X_train, y_train, event_train)

        # Restore
        cfg.M2_M3_ALPHA_RANGE = original_alpha
        cfg.M2_M3_L1_RATIO_RANGE = original_l1

        assert best_alpha is not None, "Should return tuned alpha"
        assert best_l1 is not None, "Should return tuned l1_ratio"

        # Test fit and predict
        model.fit(X_train, y_train, event_train, alpha=best_alpha, l1_ratio=best_l1)
        risk_pred = model.predict_risk(X_test)

        assert len(risk_pred) == 20, "Should predict for all test samples"

        print(f"  PASS: M2 Coxnet tuning working")
        print(f"    Best alpha: {best_alpha}, Best l1_ratio: {best_l1:.2f}")
        return True

    except Exception as e:
        print(f"  FAIL: M2 tuning error - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metric computation."""
    print("\n[5/6] Testing metrics computation...")

    try:
        from prognostic_engine.metrics import (
            harrell_c_index, uno_c_index, time_dependent_auc,
            compute_brier_score, compute_calibration
        )

        np.random.seed(42)
        n = 100

        y_time = np.random.exponential(24, n)
        y_event = np.random.binomial(1, 0.35, n)
        risk_scores = np.random.randn(n)

        # Test Harrell C-index
        harrell_c = harrell_c_index(y_time, y_event, risk_scores)
        assert 0 <= harrell_c <= 1, "C-index should be between 0 and 1"

        # Test Uno C-index
        uno_c = uno_c_index(y_time, y_event, risk_scores)
        assert 0 <= uno_c <= 1, "Uno C-index should be between 0 and 1"

        # Test time-dependent AUC
        auc_results = time_dependent_auc(y_time, y_event, risk_scores, times=[12, 36, 60])
        assert 'auc_12m' in auc_results, "Should have AUC at 12 months"
        assert 'auc_36m' in auc_results, "Should have AUC at 36 months"
        assert 'auc_60m' in auc_results, "Should have AUC at 60 months"

        # Test Brier score
        survival_probs = np.random.uniform(0.2, 0.9, n)  # Mock survival probs
        brier_results = compute_brier_score(y_time, y_event, survival_probs, times=[12, 36, 60])
        assert 'brier_36m' in brier_results, "Should have Brier at 36 months"
        assert 'ibs' in brier_results, "Should have IBS"

        # Test calibration
        cal_results = compute_calibration(y_time, y_event, survival_probs, time_point=36)
        assert 'calibration_slope' in cal_results, "Should have calibration slope"

        print("  PASS: All metrics computed correctly")
        print(f"    Harrell C: {harrell_c:.3f}")
        print(f"    Uno C: {uno_c:.3f}")
        print(f"    AUC 12/36/60m: {auc_results['auc_12m']:.3f}/{auc_results['auc_36m']:.3f}/{auc_results['auc_60m']:.3f}")
        print(f"    IBS: {brier_results['ibs']:.3f}")
        return True

    except Exception as e:
        print(f"  FAIL: Metrics error - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bootstrap():
    """Test bootstrap comparison."""
    print("\n[6/6] Testing bootstrap comparison...")

    try:
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        # Create mock predictions DataFrame
        np.random.seed(42)
        cases = [f'CASE_{i:03d}' for i in range(50)]

        predictions = []
        for case_id in cases:
            for model in ['M1_clinical_cox', 'M2_gene_elasticnet']:
                predictions.append({
                    'case_id': case_id,
                    'model': model,
                    'risk_score': np.random.randn(),
                    'survival_months': np.random.exponential(24),
                    'event': np.random.binomial(1, 0.35)
                })

        pred_df = pd.DataFrame(predictions)

        # Run bootstrap (reduced iterations for speed)
        result = patient_level_paired_bootstrap(
            pred_df,
            n_iterations=100,  # Reduced for speed
            seed=456
        )

        assert 'ci_lower' in result, "Should have CI bounds"
        assert 'p_value' in result, "Should have p-value"
        assert 'mean_diff' in result, "Should have mean difference"

        print("  PASS: Bootstrap comparison working")
        print(f"    Mean diff: {result['mean_diff']:.4f}")
        print(f"    95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"    P-value: {result['p_value']:.4f}")
        return True

    except Exception as e:
        print(f"  FAIL: Bootstrap error - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 70)
    print("PHASE 3A SMOKE TEST v2 - FORMAL NESTED CV")
    print("=" * 70)
    print("\nTesting all components of the formal nested-CV implementation")
    print("Required for full 5x5x5 nested CV training\n")

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Inner Splits", test_inner_splits()))
    results.append(("M2 Tuning", test_m2_tuning()))
    results.append(("Metrics", test_metrics()))
    results.append(("Bootstrap", test_bootstrap()))

    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:<20} {status}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n>>> SMOKE TEST v2 PASSED <<<")
        print("Ready to execute formal 5x5x5 nested CV training.")
        return 0
    else:
        print("\n>>> SMOKE TEST v2 FAILED <<<")
        print("Fix failures before proceeding to full training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
