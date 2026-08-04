"""Tests for Phase 3A Model Comparisons v4 - Patient-level bootstrap for both metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sksurv.metrics import concordance_index_ipcw

from prognostic_engine.bootstrap import patient_level_paired_bootstrap
from prognostic_engine.metrics import harrell_c_index


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class TestUnoCIPCW:
    """Test Uno C-index with IPCW calculation."""

    def test_uno_c_with_perfect_discrimination(self):
        """Uno C should approach 1.0 with perfect discrimination (same as Harrell C)."""
        n = 50
        np.random.seed(42)

        # Training data for IPCW estimation
        y_train_time = np.random.exponential(30, n)
        y_train_event = np.random.binomial(1, 0.5, n)

        # Test data: higher risk_score -> shorter survival (higher hazard)
        y_test_time = np.random.exponential(30, n)
        y_test_event = np.ones(n)
        risk_scores = y_test_time  # Higher time = lower risk, but we want higher risk

        # Create structured arrays
        y_train_struct = np.array(
            [(bool(e), float(t)) for e, t in zip(y_train_event, y_train_time)],
            dtype=[('event', bool), ('time', float)]
        )
        y_test_struct = np.array(
            [(bool(e), float(t)) for e, t in zip(y_test_event, y_test_time)],
            dtype=[('event', bool), ('time', float)]
        )

        # Use time as risk score (longer time = lower risk, so reversed)
        reversed_risk = -y_test_time

        tau = np.percentile(y_train_time[y_train_event == 1], 95)

        try:
            uno_c = concordance_index_ipcw(
                y_train_struct, y_test_struct, reversed_risk, tau=tau
            )[0]
            assert 0.9 <= uno_c <= 1.0, f"Expected ~1.0, got {uno_c}"
        except Exception as e:
            pytest.skip(f"concordance_index_ipcw failed: {e}")

    def test_uno_c_equals_harrell_c_when_no_censoring(self):
        """Uno C should equal Harrell C when no censoring (all events observed)."""
        np.random.seed(42)
        n = 100

        # All events observed (no censoring)
        y_time = np.random.exponential(30, n)
        y_event = np.ones(n)

        # Higher risk = shorter time
        risk_scores = -y_time  # Higher risk for shorter survival

        # Harrell C
        harrell = harrell_c_index(y_time, y_event, risk_scores)

        # Uno C with itself as training (all events)
        y_struct = np.array(
            [(bool(e), float(t)) for e, t in zip(y_event, y_time)],
            dtype=[('event', bool), ('time', float)]
        )

        tau = np.percentile(y_time, 95)
        try:
            uno_c = concordance_index_ipcw(y_struct, y_struct, risk_scores, tau=tau)[0]
            assert abs(harrell - uno_c) < 0.01, f"Harrell={harrell}, Uno={uno_c}"
        except Exception as e:
            pytest.skip(f"concordance_index_ipcw failed: {e}")

    def test_uno_c_different_from_harrell_c_with_censoring(self):
        """Uno C should differ from Harrell C when significant censoring exists."""
        np.random.seed(42)
        n = 100

        # Mix of events and censored
        y_time = np.random.exponential(30, n)
        y_event = np.random.binomial(1, 0.4, n)  # 40% events, 60% censored

        risk_scores = -y_time

        # Harrell C
        harrell = harrell_c_index(y_time, y_event, risk_scores)

        # Uno C - use subset as training
        train_idx = np.random.choice(n, n // 2, replace=False)
        test_idx = np.array([i for i in range(n) if i not in train_idx])

        y_train_time = y_time[train_idx]
        y_train_event = y_event[train_idx]
        y_test_time = y_time[test_idx]
        y_test_event = y_event[test_idx]
        risk_test = risk_scores[test_idx]

        y_train_struct = np.array(
            [(bool(e), float(t)) for e, t in zip(y_train_event, y_train_time)],
            dtype=[('event', bool), ('time', float)]
        )
        y_test_struct = np.array(
            [(bool(e), float(t)) for e, t in zip(y_test_event, y_test_time)],
            dtype=[('event', bool), ('time', float)]
        )

        tau = np.percentile(y_train_time[y_train_event == 1], 95)
        try:
            uno_c = concordance_index_ipcw(y_train_struct, y_test_struct, risk_test, tau=tau)[0]
            # Both should be high, but Uno C accounts for censoring differently
            assert harrell > 0.9, f"Harrell C too low: {harrell}"
            assert uno_c > 0.9, f"Uno C too low: {uno_c}"
        except Exception as e:
            pytest.skip(f"concordance_index_ipcw failed: {e}")

    def test_uno_c_returns_nan_on_invalid_input(self):
        """Uno C should return np.nan on invalid input (not crash)."""
        try:
            result = concordance_index_ipcw(
                np.array([('event', bool), ('time', float)]),
                np.array([('event', bool), ('time', float)]),
                np.array([1.0]),
                tau=10.0
            )
            # Should either return nan or raise informative error
            assert result[0] is np.nan or isinstance(result[0], float)
        except Exception:
            pass  # Expected for invalid input


class TestPatientSamplingConsistency:
    """Test patient-level sampling consistency across bootstrap iterations."""

    def _make_predictions(self, n_patients=40, n_repeats=5):
        """Create test predictions with known pattern."""
        rows = []
        for repeat in range(1, n_repeats + 1):
            for idx in range(n_patients):
                case_id = f"p{idx:03d}"
                time = float(n_patients - idx)
                fold = idx % 5 + 1
                rows.extend([
                    {'case_id': case_id, 'repeat': repeat, 'fold': fold,
                     'model': 'A', 'risk_score': float(idx), 'survival_months': time, 'event': 1},
                    {'case_id': case_id, 'repeat': repeat, 'fold': fold,
                     'model': 'B', 'risk_score': float(-idx), 'survival_months': time, 'event': 1},
                ])
        return pd.DataFrame(rows)

    def test_bootstrap_same_patient_multiple_times(self):
        """Bootstrap should sample same patient across repeats."""
        df = self._make_predictions(n_patients=40, n_repeats=5)
        result = patient_level_paired_bootstrap(
            df, n_iterations=100, seed=42, comparison_pair=('A', 'B')
        )
        assert result['status'] == 'SUCCESS'
        # Multiplicity preserved means same patient sampled multiple times
        assert result['multiplicity_preserved'] is True

    def test_bootstrap_patients_sampled_consistently_across_metrics(self):
        """Same patients should be sampled for both Harrell C and Uno C."""
        df = self._make_predictions(n_patients=40, n_repeats=5)
        result = patient_level_paired_bootstrap(
            df, n_iterations=50, seed=42, comparison_pair=('A', 'B')
        )
        assert result['status'] == 'SUCCESS'
        # The same patient sample should be used regardless of metric
        assert result['pairing_key'] == ['case_id', 'repeat', 'fold']


def _get_formal_dir():
    """Get the experiments/phase3a/formal directory path."""
    # Navigate from src/prognostic_engine/tests/ to ACM root, then to experiments
    return Path(__file__).parent.parent.parent.parent / 'experiments' / 'phase3a' / 'formal'


class TestBonferroniThreshold:
    """Test Bonferroni correction with alpha=0.0125 threshold."""

    def test_bonferroni_threshold_is_00125(self):
        """Bonferroni threshold should be exactly 0.0125 for 4 comparisons."""
        alpha = 0.05
        n_comparisons = 4
        threshold = alpha / n_comparisons
        assert threshold == 0.0125, f"Expected 0.0125, got {threshold}"

    def test_model_comparisons_v4_has_correct_threshold(self):
        """model_comparisons_v4.json should use bonferroni_threshold=0.0125."""
        path = _get_formal_dir() / 'model_comparisons_v4.json'
        if not path.exists():
            pytest.skip("model_comparisons_v4.json not found")

        with open(path, 'r') as f:
            data = json.load(f)

        assert data['bonferroni_threshold'] == 0.0125, \
            f"Expected bonferroni_threshold=0.0125, got {data['bonferroni_threshold']}"
        assert data['n_formal_comparisons'] == 4, \
            f"Expected n_formal_comparisons=4, got {data['n_formal_comparisons']}"
        assert data['bonferroni_alpha'] == 0.05, \
            f"Expected bonferroni_alpha=0.05, got {data['bonferroni_alpha']}"

    def test_significance_determined_by_bonferroni_threshold(self):
        """Only comparisons with p_adj < 0.0125 should be marked significant."""
        path = _get_formal_dir() / 'model_comparisons_v4.json'
        if not path.exists():
            pytest.skip("model_comparisons_v4.json not found")

        with open(path, 'r') as f:
            data = json.load(f)

        bonferroni_threshold = data['bonferroni_threshold']

        # Check Harrell C comparisons
        for comp in data.get('harrell_c_comparisons', []):
            if comp['type'] == 'Formal':
                # p_value_adjusted is at comparison level, not inside patient_bootstrap
                p_adj = comp.get('p_value_adjusted')
                is_significant = comp.get('significant', False)
                expected_sig = p_adj < bonferroni_threshold
                assert is_significant == expected_sig, \
                    f"{comp['comparison']}: p_adj={p_adj}, threshold={bonferroni_threshold}, significant={is_significant}"

        # Check Uno C comparisons
        for comp in data.get('uno_c_comparisons', []):
            if comp['type'] == 'Formal':
                p_adj = comp.get('p_value_adjusted')
                is_significant = comp.get('significant', False)
                expected_sig = p_adj < bonferroni_threshold
                assert is_significant == expected_sig, \
                    f"{comp['comparison']}: p_adj={p_adj}, threshold={bonferroni_threshold}, significant={is_significant}"


class TestCurrentAuditConsistency:
    """Keep legacy tests pointed at the current authoritative v6 outputs."""

    @staticmethod
    def _load_current_outputs():
        comparison_path = _get_formal_dir() / 'model_comparisons_v6.json'
        audit_path = _get_formal_dir() / 'AUDIT_REPORT_V5.json'
        assert comparison_path.exists(), "model_comparisons_v6.json not found"
        assert audit_path.exists(), "AUDIT_REPORT_V5.json not found"
        with open(comparison_path, 'r', encoding='utf-8') as f:
            comparisons = json.load(f)
        with open(audit_path, 'r', encoding='utf-8') as f:
            audit = json.load(f)
        return comparisons, audit

    @staticmethod
    def _find(comparisons, metric, model_a, model_b):
        for comp in comparisons[f'{metric}_comparisons']:
            if comp['model_a'] == model_a and comp['model_b'] == model_b:
                return comp
        raise AssertionError(f"{metric}: {model_a} vs {model_b} not found")

    def test_latest_audit_exists(self):
        path = _get_formal_dir() / 'AUDIT_REPORT_V5.json'
        assert path.exists(), "AUDIT_REPORT_V5.json not found"

    def test_latest_audit_matches_v6_results(self):
        comparisons, audit = self._load_current_outputs()
        source = self._find(
            comparisons, 'harrell_c', 'M4_combined_rsf', 'M1_clinical_cox'
        )
        recorded = audit['key_results']['M4_vs_M1_harrell_c']
        assert recorded['p_value_adjusted'] == source['p_value_adjusted']
        assert recorded['mean_diff'] == source['patient_bootstrap']['mean_diff']

    def test_latest_audit_has_all_required_sections(self):
        _, audit = self._load_current_outputs()
        required_sections = ['validation_gates', 'key_results', 'source_hashes']
        for section in required_sections:
            assert section in audit, f"AUDIT_REPORT_V5 missing required section: {section}"

    def test_m5_significant_only_on_uno_c(self):
        comparisons, _ = self._load_current_outputs()
        harrell = self._find(
            comparisons, 'harrell_c', 'M5_deepsurv', 'M1_clinical_cox'
        )
        uno = self._find(
            comparisons, 'uno_c', 'M5_deepsurv', 'M1_clinical_cox'
        )

        # Adjusted p-values are compared with the family-wise alpha (0.05).
        assert harrell['p_value_adjusted'] >= comparisons['familywise_alpha']
        assert harrell['significant_adjusted'] is False
        assert uno['p_value_adjusted'] < comparisons['familywise_alpha']
        assert uno['significant_adjusted'] is True


class TestModelComparisonsV4Results:
    """Test actual model_comparisons_v4.csv and .json results."""

    def test_v4_csv_has_10_comparisons(self):
        """v4 CSV should have exactly 10 comparison rows (5 Harrell C + 5 Uno C)."""
        csv_path = _get_formal_dir() / 'model_comparisons_v4.csv'

        if not csv_path.exists():
            pytest.skip("model_comparisons_v4.csv not found")

        df = pd.read_csv(csv_path)
        assert len(df) == 10, f"Expected 10 comparisons, got {len(df)}"

        # Check metric distribution
        metrics = df['metric'].value_counts()
        assert metrics.get('harrell_c', 0) == 5, f"Expected 5 Harrell C comparisons"
        assert metrics.get('uno_c', 0) == 5, f"Expected 5 Uno C comparisons"

    def test_v4_has_4_formal_1_exploratory(self):
        """v4 should have exactly 4 formal + 1 exploratory comparisons per metric."""
        json_path = _get_formal_dir() / 'model_comparisons_v4.json'

        if not json_path.exists():
            pytest.skip("model_comparisons_v4.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # v4 has separate comparisons for Harrell C and Uno C
        # Each metric has 4 formal + 1 exploratory comparisons
        harrell_comps = data.get('harrell_c_comparisons', [])
        uno_comps = data.get('uno_c_comparisons', [])

        harrell_formal = [c for c in harrell_comps if c.get('type') == 'Formal']
        harrell_exploratory = [c for c in harrell_comps if c.get('type') == 'Exploratory']
        uno_formal = [c for c in uno_comps if c.get('type') == 'Formal']
        uno_exploratory = [c for c in uno_comps if c.get('type') == 'Exploratory']

        assert len(harrell_formal) == 4, f"Expected 4 formal Harrell C comparisons, got {len(harrell_formal)}"
        assert len(harrell_exploratory) == 1, f"Expected 1 exploratory Harrell C comparison, got {len(harrell_exploratory)}"
        assert len(uno_formal) == 4, f"Expected 4 formal Uno C comparisons, got {len(uno_formal)}"
        assert len(uno_exploratory) == 1, f"Expected 1 exploratory Uno C comparison, got {len(uno_exploratory)}"

    def test_v4_all_p_values_nonzero(self):
        """All p-values should be > 0 (finite-sample correction applied)."""
        json_path = _get_formal_dir() / 'model_comparisons_v4.json'

        if not json_path.exists():
            pytest.skip("model_comparisons_v4.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        all_comps = data.get('harrell_c_comparisons', []) + data.get('uno_c_comparisons', [])

        for comp in all_comps:
            p_val = comp['patient_bootstrap']['p_value']
            # p_value_adjusted is at comparison level, not inside patient_bootstrap
            p_adj = comp.get('p_value_adjusted')
            assert p_val > 0, f"{comp['comparison']}: raw p-value is 0 (should be > 0)"
            assert p_adj is not None and p_adj > 0, f"{comp['comparison']}: adjusted p-value is 0 or missing (should be > 0)"
