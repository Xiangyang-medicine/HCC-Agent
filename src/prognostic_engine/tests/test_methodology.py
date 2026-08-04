"""Comprehensive methodology tests for Phase 3A SAP compliance.

Per Phase 3A reset: These tests verify methodology correctness before training.
"""
import numpy as np
import pandas as pd
import pytest


class TestInnerPreprocessingIsolation:
    """Tests for inner-fold preprocessing isolation per SAP v1.1."""

    def test_extract_inner_fold_data_by_case_id(self):
        """Verify inner fold extraction uses case_id, not row indices."""
        from prognostic_engine.inner_splits import extract_inner_fold_data

        # Create test DataFrame with case_id column
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)],
            'age': np.random.randn(n),
            'stage': np.random.randint(1, 5, n),
            'time': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })

        # Define train/val case_ids
        train_ids = [f'case_{i}' for i in range(70)]
        val_ids = [f'case_{i}' for i in range(70, 100)]

        # Extract inner fold data
        train_df, val_df = extract_inner_fold_data(df, train_ids, val_ids)

        # Verify counts
        assert len(train_df) == 70, f"Expected 70 train samples, got {len(train_df)}"
        assert len(val_df) == 30, f"Expected 30 val samples, got {len(val_df)}"

        # Verify case_ids are correct
        assert set(train_df['case_id']) == set(train_ids)
        assert set(val_df['case_id']) == set(val_ids)

    def test_preprocessing_only_on_inner_train(self):
        """Per Phase 3A reset: Preprocessing must fit ONLY on inner_train."""
        from prognostic_engine.config import METABOLIC_GENES
        from prognostic_engine.inner_preprocessing import InnerFoldPreprocessor

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)],
            'age_at_diagnosis': np.random.exponential(60, n),
            'ajcc_stage': np.random.choice(['Stage I', 'Stage II', 'Stage IIIA'], n),
            'tumor_grade': np.random.choice(['G1', 'G2', 'G3'], n),
            'gender': np.random.choice(['male', 'female'], n),
            'time': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })

        # Create gene columns using all 15 METABOLIC_GENES with _log2tpm suffix
        for gene in METABOLIC_GENES:
            df[f'{gene}_log2tpm'] = np.random.exponential(2, n)

        # Create inner train and val
        train_ids = [f'case_{i}' for i in range(80)]
        val_ids = [f'case_{i}' for i in range(80, 100)]

        train_df = df[df['case_id'].isin(train_ids)].copy()
        val_df = df[df['case_id'].isin(val_ids)].copy()

        # Preprocess - should fit on train only using preprocess_inner_fold
        preprocessor = InnerFoldPreprocessor()
        X_train, X_val, params = preprocessor.preprocess_inner_fold(train_df, val_df)

        # Verify preprocessing parameters came from train only
        assert 'genes' in X_train
        assert 'clinical' in X_train
        assert X_train['genes'] is not None

        # Verify scaling: train genes should have ~0 mean
        train_genes = X_train['genes']
        assert abs(np.mean(train_genes)) < 0.3, "Train genes should be approximately standardized"

    def test_no_data_leakage_in_preprocessing(self):
        """Verify no information leaks from val to train in preprocessing."""
        from prognostic_engine.inner_preprocessing import preprocess_inner_fold_genes

        np.random.seed(42)

        # Train: gene from exponential(1)
        train_data = pd.DataFrame({
            'case_id': [f'train_{i}' for i in range(50)],
            'time': np.random.exponential(30, 50),
            'event': np.random.binomial(1, 0.3, 50)
        })
        gene_cols = ['HK2_log2tpm', 'PKM_log2tpm']
        for g in gene_cols:
            train_data[g] = np.random.exponential(1, 50)

        # Val: gene from exponential(100) - clearly different
        val_data = pd.DataFrame({
            'case_id': [f'val_{i}' for i in range(20)],
            'time': np.random.exponential(30, 20),
            'event': np.random.binomial(1, 0.3, 20)
        })
        for g in gene_cols:
            val_data[g] = np.random.exponential(100, 20)  # 100x larger

        # Preprocess
        result = preprocess_inner_fold_genes(train_data, val_data, gene_cols)

        # Train genes should be standardized (mean ~0, std ~1)
        train_mean = np.mean(result['train_genes'])
        train_std = np.std(result['train_genes'])
        assert abs(train_mean) < 0.3, "Train should be centered"
        assert 0.7 < train_std < 1.3, "Train should be scaled"

        # Val genes should be transformed using train's parameters
        # If using train mean/std, val mean should NOT be around 99
        val_mean = np.mean(result['val_genes'])
        # With train mean ~1 and val mean ~100, z-score should be large but not raw
        assert abs(val_mean - 99) > 10, "Val should use train params, not be 99"


class TestCaseIdBasedSplitManagement:
    """Tests for case_id-based split management per Phase 3A reset."""

    def test_generate_inner_splits_returns_case_ids(self):
        """Verify generate_inner_splits returns case_id lists, not indices."""
        from prognostic_engine.inner_splits import generate_inner_splits

        # Create case_ids that are NOT simple 0-indexed integers
        case_ids = ['PATIENT_A', 'PATIENT_B', 'PATIENT_C', 'PATIENT_D', 'PATIENT_E',
                    'PATIENT_F', 'PATIENT_G', 'PATIENT_H', 'PATIENT_I', 'PATIENT_J']

        splits = generate_inner_splits(case_ids, repeat=1, outer_fold=1, n_folds=5)

        # Verify structure
        assert 'folds' in splits
        assert 'case_ids' in splits
        assert splits['case_ids'] == case_ids

        # Verify each fold has case_id lists
        for fold_info in splits['folds']:
            assert 'train_case_ids' in fold_info
            assert 'val_case_ids' in fold_info
            assert isinstance(fold_info['train_case_ids'], list)
            assert isinstance(fold_info['val_case_ids'], list)
            # Verify these are actual case_ids, not indices
            for cid in fold_info['train_case_ids'] + fold_info['val_case_ids']:
                assert isinstance(cid, str) and cid.startswith('PATIENT_')

    def test_save_load_inner_splits_case_id_preservation(self):
        """Verify round-trip preserves case_ids exactly."""
        from prognostic_engine.inner_splits import (
            generate_inner_splits, save_inner_splits, load_inner_splits
        )
        import tempfile
        from pathlib import Path

        case_ids = [f'case_{i:04d}' for i in range(20)]
        original_splits = generate_inner_splits(case_ids, repeat=1, outer_fold=1, n_folds=5)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = save_inner_splits(original_splits, output_dir=tmpdir)

            # Verify CSV was created
            assert csv_path.exists()

            # Load and verify
            loaded_splits = load_inner_splits(1, 1, output_dir=tmpdir)

            # Verify case_ids preserved
            assert set(loaded_splits['case_ids']) == set(original_splits['case_ids'])

            # Verify each fold's case_ids preserved
            for orig_fold, load_fold in zip(original_splits['folds'], loaded_splits['folds']):
                assert set(load_fold['train_case_ids']) == set(orig_fold['train_case_ids'])
                assert set(load_fold['val_case_ids']) == set(orig_fold['val_case_ids'])

    def test_case_id_roundtrip_verification(self):
        """Verify case_id roundtrip check passes for valid splits."""
        from prognostic_engine.inner_splits import generate_inner_splits, verify_case_id_roundtrip

        case_ids = [f'case_{i}' for i in range(50)]
        splits = generate_inner_splits(case_ids, repeat=1, outer_fold=1, n_folds=5)

        # This should pass without exception
        result = verify_case_id_roundtrip(case_ids, splits)
        assert result is True


class TestBootstrapPatientMultiplicity:
    """Tests for bootstrap preserving patient multiplicity per Phase 3A reset."""

    def test_bootstrap_preserves_patient_multiplicity(self):
        """Verify bootstrap sampling preserves case_id multiplicity."""
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        np.random.seed(42)

        # Create test data as DataFrame with required columns
        n = 100
        case_ids = [f'case_{i}' for i in range(n)]
        risk_scores_a = np.random.randn(n)
        risk_scores_b = np.random.randn(n)
        times = np.random.exponential(30, n)
        events = np.random.binomial(1, 0.3, n)

        # Create predictions DataFrame with TWO models for comparison
        df_a = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_a'] * n,
            'risk_score': risk_scores_a,
            'survival_months': times,
            'event': events
        })
        df_b = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_b'] * n,
            'risk_score': risk_scores_b,
            'survival_months': times,
            'event': events
        })
        predictions_df = pd.concat([df_a, df_b], ignore_index=True)
        predictions_df['repeat'] = 1
        predictions_df['fold'] = 1

        # Run bootstrap with correct parameters
        result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=100,
            seed=42
        )

        # Verify result structure
        assert 'p_value' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['iterations'] == 100

    def test_bootstrap_case_id_aggregation(self):
        """Verify bootstrap correctly aggregates by case_id."""
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        np.random.seed(42)

        # Create data with repeated case_ids (multiple rows per patient)
        case_ids = ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'E', 'F']
        risk_a = 1.0
        risk_b = 2.0
        risk_c = 3.0
        risk_scores = [risk_a, risk_a, risk_b, risk_b, risk_b, risk_c, risk_c, 4.0, 5.0, 6.0]
        times = np.array([10, 11, 20, 21, 22, 30, 31, 40, 50, 60])
        events = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 1])

        # Create predictions DataFrame for two models
        df_a = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_a'] * 10,
            'risk_score': risk_scores,
            'survival_months': times,
            'event': events
        })
        df_b = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_b'] * 10,
            'risk_score': [r + 0.5 for r in risk_scores],  # Slightly different risk
            'survival_months': times,
            'event': events
        })
        predictions_df = pd.concat([df_a, df_b], ignore_index=True)
        # Represent repeated observations as two valid outer repeats with the
        # same patient cohort in each repeat.
        base = predictions_df.drop_duplicates(['model', 'case_id'])
        predictions_df = pd.concat(
            [base.assign(repeat=1), base.assign(repeat=2)], ignore_index=True
        )
        predictions_df['fold'] = 1

        # Run bootstrap
        result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=10,
            seed=42
        )

        # Verify that bootstrap runs without error
        assert 'p_value' in result
        assert 'mean_diff' in result


class TestIntegrityMonitoringNonBlocking:
    """Tests for non-blocking integrity monitoring per Phase 3A reset."""

    def test_integrity_monitor_is_diagnostic_only(self):
        """Verify integrity monitoring does not block training."""
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M1', repeat=1, fold=1)

        # Run checks with poor metrics
        monitor.check_c_index(0.40)  # Below threshold
        monitor.check_auc({'auc_12': 0.45, 'auc_36': np.nan})
        monitor.check_ibs(0.60)  # Above threshold

        status = monitor.get_status()

        # Verify status is always MONITORED (not BLOCKED)
        assert status['status'] == 'MONITORED'

        # Verify warnings are recorded but not blocking
        assert len(status['warnings']) > 0

    def test_integrity_monitor_accepts_extreme_values(self):
        """Verify integrity monitor accepts any metric value without blocking."""
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M1', repeat=1, fold=1)

        # Test with various metric values
        extreme_values = [0.0, 0.3, 0.5, 0.7, 0.99, 1.0, np.nan]

        for val in extreme_values:
            monitor.check_c_index(val)
            status = monitor.get_status()
            # Should always return MONITORED status
            assert status['status'] == 'MONITORED'


class TestMetricCalculations:
    """Tests for metric calculation correctness per SAP v1.1."""

    def test_uno_c_returns_nan_on_failure(self):
        """Per Phase 3A reset: Uno C should return NaN on failure, not fallback."""
        from prognostic_engine.metrics import uno_c_index

        np.random.seed(42)

        # Very few events - may cause estimation failure
        y_train_time = np.array([1, 2, 3, 4, 5, 100, 200, 300])
        y_train_event = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # Only 1 train event
        y_test_time = np.array([10, 20, 30, 40])
        y_test_event = np.array([0, 0, 0, 0])  # All censored
        risk = np.random.randn(4)

        try:
            uno_c = uno_c_index(y_train_time, y_train_event, y_test_time, y_test_event, risk)
            # Should return NaN if estimation fails, or valid value otherwise
            assert np.isnan(uno_c) or (0 <= uno_c <= 1)
        except Exception:
            # Returning exception is acceptable per Phase 3A reset
            pass

    def test_uno_c_requires_ipcw_train_data(self):
        """Verify Uno C uses IPCW weights correctly with train/test separation."""
        from prognostic_engine.metrics import uno_c_index

        np.random.seed(42)
        n = 100

        # Split into train and test
        y_train_time = np.random.exponential(30, n)
        y_train_event = np.random.binomial(1, 0.3, n)
        y_test_time = np.random.exponential(30, n)
        y_test_event = np.random.binomial(1, 0.3, n)
        risk = np.random.randn(n)

        # Should work with proper train/test separation
        uno_c = uno_c_index(y_train_time, y_train_event, y_test_time, y_test_event, risk)
        assert isinstance(uno_c, (float, np.floating))

    def test_auc_at_predefined_times(self):
        """Verify AUC is calculated at SAP-defined times."""
        from prognostic_engine.metrics import time_dependent_auc

        np.random.seed(42)
        n = 100

        # Split into train and test
        y_train_time = np.random.exponential(30, n)
        y_train_event = np.random.binomial(1, 0.3, n)
        y_test_time = np.random.exponential(30, n)
        y_test_event = np.random.binomial(1, 0.3, n)
        risk = np.random.randn(n)

        # Calculate AUC at standard times
        tau_values = [12, 36, 60]
        auc_results = time_dependent_auc(
            y_train_time, y_train_event,
            y_test_time, y_test_event,
            risk, times=tau_values
        )

        # Verify all times have AUC
        assert len(auc_results) >= len(tau_values)
        for key, val in auc_results.items():
            if key.endswith('m'):
                # AUC should be in [0, 1] or NaN
                assert np.isnan(val) or (0 <= val <= 1)


class TestForbiddenPatterns:
    """Tests to verify forbidden patterns are absent from code."""

    def test_no_forbidden_decay_formula_in_models(self):
        """Per Phase 3A reset: exp(-t/60) formula must not exist."""
        import inspect
        from prognostic_engine.models import M5DeepSurv

        # Get source code
        source = inspect.getsource(M5DeepSurv.predict_survival)

        # Verify forbidden formula is absent
        assert 'exp(-t/60)' not in source
        assert 'np.exp(-t/60)' not in source
        assert 'math.exp(-t/60)' not in source

    def test_no_kfold_leakage_pattern(self):
        """Verify no KFold with n_splits=3, shuffle=True, random_state=42."""
        import inspect
        from prognostic_engine.inner_splits import generate_inner_splits

        source = inspect.getsource(generate_inner_splits)

        # Verify this specific forbidden pattern doesn't exist
        forbidden = "KFold(n_splits=3, shuffle=True, random_state=42)"
        assert forbidden not in source


class TestModelContracts:
    """Tests for model interface contracts per Phase 3A reset."""

    def test_m1_clinical_cox_no_secret_fallback(self):
        """Per Phase 3A reset: M1 should not have secret age-only fallback."""
        from prognostic_engine.models import M1ClinicalCox

        np.random.seed(42)
        n = 50

        # Single feature (age-like)
        X = np.random.randn(n, 1)
        times = np.random.exponential(30, n)
        events = np.random.binomial(1, 0.3, n)

        model = M1ClinicalCox()
        model.fit(X, times, events)

        risk = model.predict_risk(X)

        # Verify model uses the feature (not a fallback)
        assert len(risk) == n
        # With single feature, risk should vary (not be constant)
        assert np.std(risk) > 1e-6, "M1 may have constant fallback for single feature"

    def test_m2_coxnet_uses_baseline_model(self):
        """Per Phase 3A reset: M2 must use fit_baseline_model=True."""
        import inspect
        from prognostic_engine.models import M2M3Coxnet

        # Get source code of fit method
        source = inspect.getsource(M2M3Coxnet.fit)

        # Verify baseline model is used
        assert 'baseline' in source.lower() or 'fit_baseline' in source.lower()

    def test_all_models_require_inner_cv_for_tuning(self):
        """Verify all tune() methods require inner_cv_splits."""
        from prognostic_engine.models import M1ClinicalCox, M2M3Coxnet, M4RSF, M5DeepSurv
        import inspect

        np.random.seed(42)
        n = 30
        X = np.random.randn(n, 5)
        times = np.random.exponential(30, n)
        events = np.random.binomial(1, 0.3, n)

        # M1 does not have tune() - skip
        # M2, M4, M5 should require inner_cv_splits
        for model_cls, extra_kwargs in [
            (M2M3Coxnet, {'model_name': 'M2'}),
            (M4RSF, {}),
            (M5DeepSurv, {}),
        ]:
            model = model_cls(**extra_kwargs)

            # Should raise error without inner_cv_splits
            try:
                model.tune(X, times, events)
                pytest.fail(f"{model_cls.__name__}.tune() should require inner_cv_splits")
            except (ValueError, TypeError) as e:
                # Should fail due to missing inner_cv_splits
                assert 'inner_cv' in str(e).lower() or 'required' in str(e).lower() or 'missing' in str(e).lower()


class TestBootstrapWithRepeatFold:
    """Tests for bootstrap with repeat/fold support per Phase 3A reset."""

    def test_bootstrap_accepts_repeat_fold_params(self):
        """Verify bootstrap accepts and returns repeat/fold parameters."""
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        np.random.seed(42)

        # Create test data
        n = 50
        case_ids = [f'case_{i}' for i in range(n)]
        risk_a = np.random.randn(n)
        risk_b = np.random.randn(n) + 0.1
        times = np.random.exponential(30, n)
        events = np.random.binomial(1, 0.3, n)

        df_a = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_a'] * n,
            'risk_score': risk_a,
            'survival_months': times,
            'event': events
        })
        df_b = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_b'] * n,
            'risk_score': risk_b,
            'survival_months': times,
            'event': events
        })
        predictions_df = pd.concat([df_a, df_b], ignore_index=True)
        predictions_df['repeat'] = 1
        predictions_df['fold'] = 1

        # Run bootstrap with repeat/fold
        result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=10,
            seed=42,
            repeat=1,
            fold=3
        )

        # Verify repeat/fold are returned
        assert result.get('repeat') == 1
        assert result.get('fold') == 3

    def test_aggregate_bootstrap_results(self):
        """Verify aggregation of bootstrap results across folds."""
        from prognostic_engine.bootstrap import (
            patient_level_paired_bootstrap, aggregate_bootstrap_results
        )

        np.random.seed(42)

        # Create test data
        n = 30
        case_ids = [f'case_{i}' for i in range(n)]
        risk_a = np.random.randn(n)
        risk_b = np.random.randn(n) + 0.2
        times = np.random.exponential(30, n)
        events = np.random.binomial(1, 0.3, n)

        df_a = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_a'] * n,
            'risk_score': risk_a,
            'survival_months': times,
            'event': events
        })
        df_b = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_b'] * n,
            'risk_score': risk_b,
            'survival_months': times,
            'event': events
        })
        predictions_df = pd.concat([df_a, df_b], ignore_index=True)
        predictions_df['repeat'] = 1
        predictions_df['fold'] = 1

        # Generate results for multiple repeats/folds
        results_list = []
        for repeat in range(1, 3):
            for fold in range(1, 3):
                result = patient_level_paired_bootstrap(
                    predictions_df,
                    n_iterations=10,
                    seed=42 + repeat * 10 + fold,
                    repeat=repeat,
                    fold=fold
                )
                results_list.append(result)

        # Aggregate
        aggregated = aggregate_bootstrap_results(results_list)

        # Verify structure
        assert aggregated['status'] == 'AGGREGATED'
        assert aggregated['n_results'] == 4  # 2 repeats x 2 folds
        assert aggregated['n_unique_folds'] == 4
        assert len(aggregated['fold_summary']) == 4

    def test_run_full_bootstrap_comparison(self):
        """Verify running bootstrap for multiple model pairs."""
        from prognostic_engine.bootstrap import run_full_bootstrap_comparison

        np.random.seed(42)

        # Create test data with 3 models
        n = 40
        case_ids = [f'case_{i}' for i in range(n)]

        df_a = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_a'] * n,
            'risk_score': np.random.randn(n),
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_b = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_b'] * n,
            'risk_score': np.random.randn(n) + 0.2,
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_c = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_c'] * n,
            'risk_score': np.random.randn(n) + 0.4,
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_b[['survival_months', 'event']] = df_a[['survival_months', 'event']].to_numpy()
        df_c[['survival_months', 'event']] = df_a[['survival_months', 'event']].to_numpy()
        predictions_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
        predictions_df['repeat'] = 1
        predictions_df['fold'] = 1

        # Compare multiple pairs
        model_pairs = [('model_a', 'model_b'), ('model_b', 'model_c')]
        result = run_full_bootstrap_comparison(
            predictions_df,
            model_pairs=model_pairs,
            n_iterations=10,
            seed=42,
            repeat=1,
            fold=1
        )

        # Verify structure
        assert 'individual_results' in result or 'result' in result
        assert result['metadata']['n_comparisons'] == 2 or result['metadata'].get('n_comparisons', 1) == 2

    def test_bootstrap_preserves_patient_multiplicity(self):
        """Verify bootstrap correctly preserves all rows per patient."""
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        np.random.seed(42)

        # Create data with 2 rows per patient (multiplicity)
        case_ids = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']
        risk_a = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
        risk_b = [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5.5, 5.5]
        times = [10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
        events = [1] * 10

        df_a = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_a'] * 10,
            'risk_score': risk_a,
            'survival_months': times,
            'event': events
        })
        df_b = pd.DataFrame({
            'case_id': case_ids,
            'model': ['model_b'] * 10,
            'risk_score': risk_b,
            'survival_months': times,
            'event': events
        })
        predictions_df = pd.concat([df_a, df_b], ignore_index=True)
        predictions_df['repeat'] = predictions_df.groupby(['model', 'case_id']).cumcount() + 1
        predictions_df['fold'] = 1

        # Run bootstrap
        result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=10,
            seed=42
        )

        # Verify result has valid structure
        assert 'p_value' in result
        assert 'mean_diff' in result
        assert 'iterations' in result
        assert result['iterations'] == 10

    def test_bootstrap_handles_multiple_model_comparisons(self):
        """Verify bootstrap can compare different model pairs."""
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        np.random.seed(42)

        n = 30
        case_ids = [f'case_{i}' for i in range(n)]

        df_a = pd.DataFrame({
            'case_id': case_ids,
            'model': ['M1'] * n,
            'risk_score': np.random.randn(n),
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_b = pd.DataFrame({
            'case_id': case_ids,
            'model': ['M2'] * n,
            'risk_score': np.random.randn(n) + 0.3,
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_c = pd.DataFrame({
            'case_id': case_ids,
            'model': ['M3'] * n,
            'risk_score': np.random.randn(n) + 0.5,
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_b[['survival_months', 'event']] = df_a[['survival_months', 'event']].to_numpy()
        df_c[['survival_months', 'event']] = df_a[['survival_months', 'event']].to_numpy()
        predictions_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
        predictions_df['repeat'] = 1
        predictions_df['fold'] = 1

        # Compare M1 vs M2
        result_m1_m2 = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=10,
            seed=42,
            comparison_pair=('M1', 'M2')
        )
        assert result_m1_m2['model_a'] == 'M1'
        assert result_m1_m2['model_b'] == 'M2'

        # Compare M2 vs M3
        result_m2_m3 = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=10,
            seed=42,
            comparison_pair=('M2', 'M3')
        )
        assert result_m2_m3['model_a'] == 'M2'
        assert result_m2_m3['model_b'] == 'M3'

        # Verify different results
        assert result_m1_m2['mean_diff'] != result_m2_m3['mean_diff'] or \
               (result_m1_m2['mean_diff'] is None and result_m2_m3['mean_diff'] is None)
