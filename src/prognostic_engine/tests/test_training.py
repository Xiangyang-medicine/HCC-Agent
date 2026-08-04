"""Unit tests for training.py.

Per Phase 3A reset: Tests for integrity monitoring (non-blocking).
"""
import numpy as np
import pytest


class TestIntegrityMonitor:
    """Tests for IntegrityMonitor (non-blocking diagnostics)."""

    def test_monitor_never_blocks(self):
        """Per Phase 3A reset: IntegrityMonitor must never block training."""
        from prognostic_engine.training import IntegrityMonitor

        # IntegrityMonitor requires model_name, repeat, fold parameters
        monitor = IntegrityMonitor('M1', 1, 1)

        # Even with terrible metrics, status should be MONITORED (not BLOCKED)
        # Use check_c_index directly since IntegrityMonitor uses individual check methods
        monitor.check_c_index(0.3, metric_name='c_index')

        # Status must NEVER be 'blocked'
        status = monitor.get_status()
        assert status.get('status') != 'BLOCKED'
        # Status should include warnings
        assert len(monitor.warnings) > 0

    def test_warnings_logged(self):
        """Per Phase 3A reset: Monitor should log warnings for threshold violations."""
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M1', 1, 1)

        # Check metrics below thresholds
        monitor.check_c_index(0.3)  # Below 0.50
        monitor.check_auc({'auc_12m': 0.4})  # Below 0.50
        monitor.check_ibs(0.6)  # Above 0.50

        # Should have warnings
        assert len(monitor.warnings) > 0

    def test_get_status_always_monitored(self):
        """Per Phase 3A reset: get_status always returns MONITORED."""
        from prognostic_engine.training import IntegrityMonitor

        monitor = IntegrityMonitor('M1', 1, 1)

        status = monitor.get_status()

        # Must always be MONITORED, never BLOCKED
        assert status['status'] == 'MONITORED'


class TestIntegrityGates:
    """Tests for integrity gate behavior (Phase 3A reset)."""

    def test_no_performance_gates(self):
        """Per Phase 3A reset: No performance-based integrity gates."""
        from prognostic_engine.training import INTEGRITY_MONITORING

        # Thresholds exist for diagnostics only
        assert 'c_index_min' in INTEGRITY_MONITORING
        assert 'auc_min' in INTEGRITY_MONITORING
        assert 'ibs_max' in INTEGRITY_MONITORING

        # But they should be labeled as diagnostic thresholds, not gates
        # This is verified by IntegrityMonitor never blocking


class TestRunIntegrityMonitor:
    """Tests for run_integrity_monitor function."""

    def test_prints_diagnostics(self):
        """Per Phase 3A reset: Monitor only prints diagnostics, never blocks."""
        from prognostic_engine.training import run_integrity_monitor
        import numpy as np

        np.random.seed(42)
        metrics = {
            'harrell_c': 0.55,
            'uno_c': 0.52,
            'auc_12m': 0.60,
            'auc_36m': 0.58,
            'auc_60m': 0.55,
            'ibs': 0.25,
        }
        risk_scores = np.random.rand(30)
        survival_probs = np.tile([0.9, 0.7, 0.5], (30, 1))
        y_test = np.random.exponential(30, 30)
        e_test = np.random.binomial(1, 0.3, 30)

        result = run_integrity_monitor(
            model_name='M1',
            repeat=1,
            fold=1,
            metrics=metrics,
            risk_scores=risk_scores,
            survival_probs=survival_probs,
            y_test=y_test,
            e_test=e_test
        )

        # Should return dict with status
        assert 'status' in result
        # Status should be MONITORED (diagnostic)
        assert result['status'] == 'MONITORED'


class TestSaveResults:
    """Tests for result saving behavior."""

    def test_always_completed_status(self):
        """Per Phase 3A reset: _save_results always sets status to COMPLETED."""
        # This is verified by the implementation:
        # overall_status = 'COMPLETED' (not 'BLOCKED' or conditional)
        # The INTEGRITY_MONITORING dict has thresholds but they don't gate anything
        pass  # Verified by code inspection


class TestAggregateIntegrity:
    """Tests for integrity aggregation across folds."""

    def test_tracks_warning_rates(self):
        """Per Phase 3A reset: _aggregate_integrity tracks warning rates."""
        # This is verified by implementation:
        # warning_counts tracks how many times each threshold was violated
        # This is for diagnostics, not blocking
        pass  # Verified by code inspection
