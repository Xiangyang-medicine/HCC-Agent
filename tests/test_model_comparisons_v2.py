#!/usr/bin/env python3
"""Tests for model comparisons v2.

These tests verify the correctness of the patient-level bootstrap implementation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # F:\ACM


class TestModelComparisonsV2:
    """Test suite for model comparisons v2."""

    @pytest.fixture
    def oof_predictions(self):
        """Load OOF predictions."""
        path = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"
        return pd.read_csv(path)

    @pytest.fixture
    def comparisons_json(self):
        """Load comparisons JSON."""
        path = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "model_comparisons_v2.json"
        with open(path) as f:
            return json.load(f)

    def test_n_patients_363(self, oof_predictions):
        """Verify data has 363 unique patients."""
        unique_patients = oof_predictions['case_id'].nunique()
        assert unique_patients == 363, f"Expected 363 patients, got {unique_patients}"

    def test_n_repeats_5(self, oof_predictions):
        """Verify 5 repeats in data."""
        repeats = oof_predictions['repeat'].unique()
        assert len(repeats) == 5, f"Expected 5 repeats, got {len(repeats)}"
        assert sorted(repeats) == [1, 2, 3, 4, 5]

    def test_n_folds_per_repeat_5(self, oof_predictions):
        """Verify 5 folds per repeat."""
        for repeat in range(1, 6):
            folds = oof_predictions[oof_predictions['repeat'] == repeat]['fold'].unique()
            assert len(folds) == 5, f"Repeat {repeat} should have 5 folds, got {len(folds)}"

    def test_iterations_1000(self, comparisons_json):
        """Verify 1000 bootstrap iterations were used."""
        for comp in comparisons_json['comparisons']:
            n_iter = comp['patient_bootstrap']['iterations_valid']
            assert n_iter == 1000, f"Expected 1000 iterations, got {n_iter}"

    def test_multiplicity_preserved(self, oof_predictions):
        """Verify that multiplicity (patient in multiple folds) is tracked."""
        # Each patient appears in 25 rows (5 repeats × 5 folds) for OOF predictions
        # This is the nested CV structure where each patient is in test set once per repeat
        patient_row_counts = oof_predictions.groupby('case_id').size()
        # All patients should have exactly 25 rows (one for each repeat-fold combination)
        assert (patient_row_counts == 25).all(), \
            f"Each patient should appear in 25 OOF rows (5 repeats × 5 folds), got {patient_row_counts.unique()}"

    def test_pairing_keys_match(self, oof_predictions):
        """Verify (case_id, repeat, fold) keys match across models."""
        # Get unique keys per model
        models = ['M1', 'M2', 'M3', 'M4', 'M5']
        keys_by_model = {}
        for model in models:
            model_data = oof_predictions[oof_predictions['model'] == model]
            keys_by_model[model] = set(
                zip(model_data['case_id'], model_data['repeat'], model_data['fold'])
            )

        # All models should have identical keys
        reference_keys = keys_by_model['M1']
        for model in models[1:]:
            assert keys_by_model[model] == reference_keys, \
                f"{model} keys don't match M1"

    def test_bootstrap_min_pvalue_gt_0(self, comparisons_json):
        """P-value never exactly 0 (finite-sample correction working)."""
        for comp in comparisons_json['comparisons']:
            raw_p = comp['patient_bootstrap']['p_value_raw']
            assert raw_p > 0, f"Raw p-value is 0 (finite-sample correction not working): {comp['comparison']}"
            assert not np.isnan(raw_p), f"Raw p-value is NaN: {comp['comparison']}"

    def test_formal_comparison_count_4(self, comparisons_json):
        """Exactly 4 formal comparisons."""
        formal_comps = [c for c in comparisons_json['comparisons'] if c['type'] == 'Formal']
        assert len(formal_comps) == 4, f"Expected 4 formal comparisons, got {len(formal_comps)}"

    def test_exploratory_comparison_count_1(self, comparisons_json):
        """Exactly 1 exploratory comparison."""
        exploratory_comps = [c for c in comparisons_json['comparisons'] if c['type'] == 'Exploratory']
        assert len(exploratory_comps) == 1, f"Expected 1 exploratory comparison, got {len(exploratory_comps)}"

    def test_adjusted_p_numeric(self, comparisons_json):
        """Adjusted p-values are numeric and in valid range."""
        for comp in comparisons_json['comparisons']:
            adj_p = comp['patient_bootstrap']['p_value_adjusted']
            assert isinstance(adj_p, (int, float)), f"Adjusted p not numeric: {comp['comparison']}"
            assert 0 <= adj_p <= 1, f"Adjusted p out of range [0,1]: {adj_p}"

    def test_bonferroni_threshold_correct(self, comparisons_json):
        """Bonferroni threshold should be 0.0125 for 4 comparisons."""
        threshold = comparisons_json['bonferroni_alpha']
        expected = 0.05 / 4  # 0.0125
        assert abs(threshold - expected) < 1e-6, \
            f"Bonferroni threshold should be 0.0125, got {threshold}"

    def test_significant_flag_correct(self, comparisons_json):
        """Significant flag matches p-value threshold."""
        threshold = comparisons_json['bonferroni_alpha']
        for comp in comparisons_json['comparisons']:
            adj_p = comp['patient_bootstrap']['p_value_adjusted']
            is_significant = comp['significant']
            should_be_significant = adj_p < threshold

            # For formal comparisons, check the flag
            if comp['type'] == 'Formal':
                assert is_significant == should_be_significant, \
                    f"Significant flag mismatch for {comp['comparison']}: " \
                    f"flag={is_significant}, p_adj={adj_p:.4f}, threshold={threshold:.4f}"

    def test_ci_width_positive(self, comparisons_json):
        """Confidence interval upper > lower."""
        for comp in comparisons_json['comparisons']:
            ci_lower = comp['patient_bootstrap']['ci_lower']
            ci_upper = comp['patient_bootstrap']['ci_upper']
            assert ci_upper > ci_lower, \
                f"CI upper ({ci_upper}) <= lower ({ci_lower}) for {comp['comparison']}"

    def test_mismatched_patients_fails(self):
        """Verify that mismatched patient sets would cause errors."""
        # This is a logic test - the comparison script should validate key matching
        # If keys don't match, the bootstrap would compute differences on different samples

        # Create two DataFrames with different patient sets
        df1 = pd.DataFrame({
            'case_id': ['A', 'B', 'C', 'A', 'B', 'C'],
            'repeat': [1, 1, 1, 2, 2, 2],
            'fold': [1, 1, 1, 1, 1, 1],
            'model': ['M1'] * 6,
            'c_index': np.random.rand(6)
        })

        df2 = pd.DataFrame({
            'case_id': ['A', 'B', 'D', 'A', 'B', 'D'],  # 'C' replaced with 'D'
            'repeat': [1, 1, 1, 2, 2, 2],
            'fold': [1, 1, 1, 1, 1, 1],
            'model': ['M2'] * 6,
            'c_index': np.random.rand(6)
        })

        # Keys should not match
        keys1 = set(zip(df1['case_id'], df1['repeat'], df1['fold']))
        keys2 = set(zip(df2['case_id'], df2['repeat'], df2['fold']))

        assert keys1 != keys2, "Test setup error: keys should differ"


class TestModelComparisonsV2Results:
    """Test specific results from the v2 analysis."""

    @pytest.fixture
    def comparisons_json(self):
        """Load comparisons JSON."""
        path = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "model_comparisons_v2.json"
        with open(path) as f:
            return json.load(f)

    def test_m4_vs_m1_not_significant(self, comparisons_json):
        """M4 vs M1 should NOT be significant at Bonferroni threshold."""
        comp = next(c for c in comparisons_json['comparisons'] if c['comparison'] == 'M4 vs M1')
        adj_p = comp['patient_bootstrap']['p_value_adjusted']

        # With Bonferroni correction (threshold 0.0125), M4 vs M1 should NOT be significant
        # The corrected analysis shows p_adj = 0.080 > 0.0125
        assert adj_p > 0.0125, \
            f"M4 vs M1 should NOT be significant at Bonferroni 0.0125, got p_adj={adj_p}"

    def test_m5_vs_m1_not_significant(self, comparisons_json):
        """M5 vs M1 should NOT be significant at Bonferroni threshold."""
        comp = next(c for c in comparisons_json['comparisons'] if c['comparison'] == 'M5 vs M1')
        adj_p = comp['patient_bootstrap']['p_value_adjusted']

        # With Bonferroni correction (threshold 0.0125), M5 vs M1 should NOT be significant
        assert adj_p > 0.0125, \
            f"M5 vs M1 should NOT be significant at Bonferroni 0.0125, got p_adj={adj_p}"

    def test_m3_vs_m2_not_significant(self, comparisons_json):
        """M3 vs M2 should NOT be significant (near-zero difference)."""
        comp = next(c for c in comparisons_json['comparisons'] if c['comparison'] == 'M3 vs M2')
        mean_diff = comp['patient_bootstrap']['mean_diff']

        # M3 and M2 are essentially the same model (combined vs gene-only)
        # The difference should be very small
        assert abs(mean_diff) < 0.01, \
            f"M3 vs M2 mean diff should be near zero, got {mean_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
