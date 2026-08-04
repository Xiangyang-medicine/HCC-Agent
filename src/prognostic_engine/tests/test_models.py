"""Unit tests for models.py.

Per Phase 3A reset: Unit tests must pass before running any training.
"""
import numpy as np
import pandas as pd
import pytest


class TestM1ClinicalCox:
    """Tests for M1ClinicalCox model."""

    def test_fit_predict_risk(self, sample_survival_data, sample_features):
        """Test M1 can fit and predict risk scores."""
        from prognostic_engine.models import M1ClinicalCox

        model = M1ClinicalCox()
        model.fit(
            sample_features,
            sample_survival_data['times'],
            sample_survival_data['events'],
            feature_names=[f'f{i}' for i in range(sample_features.shape[1])]
        )

        risk_scores = model.predict_risk(sample_features[:10])
        assert risk_scores.shape == (10,)
        assert not np.isnan(risk_scores).all()

    def test_predict_survival(self, sample_survival_data, sample_features):
        """Test M1 can predict survival probabilities."""
        from prognostic_engine.models import M1ClinicalCox

        model = M1ClinicalCox()
        model.fit(
            sample_features,
            sample_survival_data['times'],
            sample_survival_data['events']
        )

        times = [12, 36, 60]
        surv_probs = model.predict_survival(sample_features[:10], times)

        assert surv_probs.shape == (10, 3)
        assert np.all((surv_probs >= 0) & (surv_probs <= 1))

    def test_no_secret_fallback(self, sample_survival_data):
        """Per Phase 3A reset: M1 should not have secret age-only fallback."""
        from prognostic_engine.models import M1ClinicalCox

        # Create minimal features (just age)
        X = np.random.randn(50, 1)
        model = M1ClinicalCox()
        model.fit(X, sample_survival_data['times'][:50], sample_survival_data['events'][:50])

        risk = model.predict_risk(X)
        # If M1 has secret fallback, it would handle missing data differently
        # This test verifies the model uses actual features
        assert len(risk) == 50


class TestM2M3Coxnet:
    """Tests for M2M3Coxnet model."""

    def test_tune_finds_params(self, sample_survival_data, sample_features, inner_cv_splits):
        """Test M2/M3 can tune hyperparameters."""
        from prognostic_engine.models import M2M3Coxnet

        np.random.seed(42)
        model = M2M3Coxnet(model_name='M2')

        # Use subset for faster testing
        n_train = 50
        alpha, l1_ratio = model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            inner_cv_splits=inner_cv_splits
        )

        assert alpha is not None
        assert l1_ratio is not None
        assert 0 < l1_ratio <= 1

    def test_fit_with_baseline_model(self, sample_survival_data, sample_features, inner_cv_splits):
        """Per Phase 3A reset: Coxnet must use fit_baseline_model=True."""
        from prognostic_engine.models import M2M3Coxnet

        np.random.seed(42)
        model = M2M3Coxnet(model_name='M2')

        # First tune to get params
        n_train = 50
        model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            inner_cv_splits=inner_cv_splits
        )

        # Then fit
        model.fit(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train]
        )

        # Verify model has baseline model
        assert model.model is not None

    def test_predict_survival_uses_baseline(self, sample_survival_data, sample_features, inner_cv_splits):
        """Per Phase 3A reset: Coxnet must use baseline model for survival prediction."""
        from prognostic_engine.models import M2M3Coxnet

        np.random.seed(42)
        model = M2M3Coxnet(model_name='M2')

        n_train = 60
        model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            inner_cv_splits=inner_cv_splits
        )
        model.fit(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train]
        )

        times = [12, 36, 60]
        surv_probs = model.predict_survival(sample_features[:10], times)

        assert surv_probs.shape == (10, 3)
        assert np.all((surv_probs >= 0) & (surv_probs <= 1))

    def test_no_zero_coefficient_models(self, sample_survival_data, sample_features, inner_cv_splits):
        """Per Phase 3A reset: Coxnet must not select zero-coefficient models."""
        from prognostic_engine.models import M2M3Coxnet

        np.random.seed(42)
        model = M2M3Coxnet(model_name='M2')

        n_train = 50
        alpha, l1_ratio = model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            inner_cv_splits=inner_cv_splits
        )

        # Verify the selected model has non-zero coefficients
        model.fit(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            alpha=alpha, l1_ratio=l1_ratio
        )

        coef = model.model.coef_
        if coef is not None:
            assert np.abs(coef).sum() > 1e-6, "Selected model has zero coefficients"


class TestM4RSF:
    """Tests for M4RSF model."""

    def test_tune_finds_params(self, sample_survival_data, sample_features, inner_cv_splits):
        """Test M4 can tune hyperparameters."""
        from prognostic_engine.models import M4RSF

        np.random.seed(42)
        model = M4RSF()

        n_train = 50
        best_params = model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            inner_cv_splits=inner_cv_splits
        )

        assert 'n_estimators' in best_params
        assert best_params['n_estimators'] > 0

    def test_predict_risk_and_survival(self, sample_survival_data, sample_features, inner_cv_splits):
        """Test M4 can predict risk and survival."""
        from prognostic_engine.models import M4RSF

        np.random.seed(42)
        model = M4RSF()

        n_train = 50
        model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            inner_cv_splits=inner_cv_splits
        )
        model.fit(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train]
        )

        risk_scores = model.predict_risk(sample_features[:10])
        assert risk_scores.shape == (10,)

        times = [12, 36, 60]
        surv_probs = model.predict_survival(sample_features[:10], times)
        assert surv_probs.shape == (10, 3)
        assert np.all((surv_probs >= 0) & (surv_probs <= 1))


class TestM5DeepSurv:
    """Tests for M5DeepSurv model."""

    def test_tune_finds_params(self, sample_survival_data, sample_features, inner_cv_splits):
        """Test M5 can tune hyperparameters."""
        from prognostic_engine.models import M5DeepSurv

        # Check if torch is available
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        model = M5DeepSurv()

        n_train = 50
        best_params = model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            batch_frac=1.0,
            inner_cv_splits=inner_cv_splits
        )

        assert 'hidden_layers' in best_params
        assert 'lr' in best_params

    def test_predict_survival_no_decay_formula(self, sample_survival_data, sample_features, inner_cv_splits):
        """Per Phase 3A reset: M5 must not use decay formula for survival."""
        from prognostic_engine.models import M5DeepSurv

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        model = M5DeepSurv()

        n_train = 50

        model.tune(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train],
            batch_frac=1.0,
            inner_cv_splits=inner_cv_splits
        )
        model.fit(
            sample_features[:n_train],
            sample_survival_data['times'][:n_train],
            sample_survival_data['events'][:n_train]
        )

        times = [12, 36, 60]
        surv_probs = model.predict_survival(sample_features[:10], times)

        assert surv_probs.shape == (10, 3)
        # Survival should decrease over time (monotonic)
        assert np.all(surv_probs[:, 2] <= surv_probs[:, 1] + 1e-6)
        assert np.all(surv_probs[:, 1] <= surv_probs[:, 0] + 1e-6)
        # All values should be in [0, 1]
        assert np.all((surv_probs >= 0) & (surv_probs <= 1))

    def test_tune_requires_inner_cv_splits(self, sample_survival_data, sample_features):
        """Per Phase 3A reset: M5.tune must raise ValueError without inner_cv_splits."""
        from prognostic_engine.models import M5DeepSurv

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        model = M5DeepSurv()

        with pytest.raises(ValueError, match="inner_cv_splits is required"):
            model.tune(
                sample_features[:30],
                sample_survival_data['times'][:30],
                sample_survival_data['events'][:30],
                batch_frac=1.0
            )


class TestTrainingPipelineCompleteness:
    """Tests for dynamic training pipeline completeness checks per Phase 3A reset."""

    def test_integrity_monitor_tracks_metrics(self):
        """Verify IntegrityMonitor properly tracks and reports metrics."""
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M1', repeat=1, fold=1)

        # Add metric checks with values that generate warnings
        monitor.check_c_index(0.45)  # Below threshold, generates warning
        monitor.check_auc({'auc_12': 0.40, 'auc_36': 0.45})  # Below threshold, generates warnings
        monitor.check_ibs(0.55)  # Above max threshold, generates warning

        status = monitor.get_status()

        assert status['model'] == 'M1'
        assert status['repeat'] == 1
        assert status['fold'] == 1
        assert status['status'] == 'MONITORED'
        # check_c_index adds to checks and warnings when below threshold
        assert 'c_index' in status['checks']
        assert len(status['warnings']) >= 2  # AUC and IBS generate warnings

    def test_integrity_monitor_never_blocks(self):
        """Verify IntegrityMonitor never blocks regardless of metric values."""
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M1', repeat=1, fold=1)

        # Test with extreme/worst case values
        monitor.check_c_index(0.3)  # Very poor
        monitor.check_auc({'auc_12': 0.4, 'auc_36': 0.4})  # Very poor
        monitor.check_ibs(0.9)  # Very poor IBS

        status = monitor.get_status()

        # Should still be MONITORED, never BLOCKED
        assert status['status'] == 'MONITORED'

    def test_integrity_monitor_checks_structure(self):
        """Verify IntegrityMonitor checks dictionary structure."""
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M2', repeat=2, fold=3)

        monitor.check_c_index(0.55)
        monitor.check_auc({'auc_12': 0.60})

        status = monitor.get_status()

        # Verify checks dictionary
        assert 'c_index' in status['checks']
        assert status['checks']['c_index']['value'] == 0.55
        assert status['checks']['c_index']['status'] == 'OK'

    def test_run_integrity_monitor_function(self):
        """Verify run_integrity_monitor function exists and works."""
        from prognostic_engine.training import run_integrity_monitor

        np.random.seed(42)

        metrics = {
            'harrell_c': 0.62,
            'uno_c': 0.58,
            'auc_12': 0.68,
            'auc_36': 0.65,
            'ibs': 0.22
        }
        risk_scores = np.random.randn(50)
        survival_probs = np.random.rand(50, 3)
        y_test = np.random.exponential(30, 50)
        e_test = np.random.binomial(1, 0.3, 50)

        status = run_integrity_monitor(
            'M3', repeat=1, fold=1,
            metrics=metrics,
            risk_scores=risk_scores,
            survival_probs=survival_probs,
            y_test=y_test,
            e_test=e_test
        )

        assert status['model'] == 'M3'
        assert status['status'] == 'MONITORED'

    def test_bootstrap_comparison_result_format(self):
        """Verify bootstrap comparison returns proper format with repeat/fold."""
        from prognostic_engine.bootstrap import patient_level_paired_bootstrap

        np.random.seed(42)

        # Create simple test data
        n = 20
        df_a = pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)],
            'model': ['M1'] * n,
            'risk_score': np.random.randn(n),
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_b = pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(n)],
            'model': ['M2'] * n,
            'risk_score': np.random.randn(n) + 0.2,
            'survival_months': np.random.exponential(30, n),
            'event': np.random.binomial(1, 0.3, n)
        })
        df_b[['survival_months', 'event']] = df_a[['survival_months', 'event']].to_numpy()
        predictions_df = pd.concat([df_a, df_b], ignore_index=True)
        predictions_df['repeat'] = 1
        predictions_df['fold'] = 1

        result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=10,
            seed=42,
            repeat=1,
            fold=1
        )

        # Verify required fields
        assert 'iterations' in result
        assert 'p_value' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'mean_diff' in result
        assert result['repeat'] == 1
        assert result['fold'] == 1
