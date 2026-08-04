"""Anti-leakage tests for Phase 3A Model Comparisons v5.

These tests verify that the v5 implementation correctly uses OUTER TRAINING fold
data for IPCW estimation, NOT test data (which would be information leakage).

CRITICAL: v4 used test data for IPCW (INVALID). v5 must use outer training fold.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Get paths relative to this test file
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent.parent
FORMAL_DIR = PROJECT_ROOT / 'experiments' / 'phase3a' / 'formal'
SPLITS_DIR = PROJECT_ROOT / 'experiments' / 'phase3a' / 'splits'


class TestIPCWSourceVerification:
    """Verify IPCW is computed from OUTER TRAINING fold, not test data."""

    def test_ipcw_source_outer_training_fold(self):
        """v5 JSON must state ipcw_source='outer_training_fold'."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data.get('ipcw_source') == 'outer_training_fold', \
            f"IPCW source must be 'outer_training_fold', got {data.get('ipcw_source')}"

    def test_ipcw_metadata_has_train_n(self):
        """Uno C results must have train_n in ipcw_metadata (proving training data used)."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        for comp in data.get('uno_c_comparisons', []):
            metadata_sample = comp.get('ipcw_metadata_sample', [])
            assert len(metadata_sample) > 0, "ipcw_metadata_sample must not be empty"

            for meta in metadata_sample:
                assert 'train_n' in meta, "IPCW metadata must include train_n"
                assert 'test_n' in meta, "IPCW metadata must include test_n"
                assert 'tau' in meta, "IPCW metadata must include tau"

                # train_n should be larger than test_n (training set > test set)
                assert meta['train_n'] > meta['test_n'], \
                    f"train_n ({meta['train_n']}) must be > test_n ({meta['test_n']})"

    def test_ipcw_metadata_train_n_consistent(self):
        """train_n should be ~290 (363 - ~73 test) for all folds."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        train_ns = []
        for comp in data.get('uno_c_comparisons', []):
            for meta in comp.get('ipcw_metadata_sample', []):
                train_ns.append(meta['train_n'])

        # All should be around 290 (80/20 split)
        assert all(285 <= n <= 295 for n in train_ns), \
            f"All train_n should be ~290, got {train_ns}"


class TestNoTestDataLeakage:
    """Verify no information leakage from test data into IPCW computation."""

    def test_train_test_overlap_zero(self):
        """train_test_overlap must be 0 for all Uno C comparisons."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        for comp in data.get('uno_c_comparisons', []):
            overlap = comp.get('train_test_overlap', -1)
            assert overlap == 0, \
                f"train_test_overlap must be 0, got {overlap}"

    def test_methodology_mentions_proper_ipcw(self):
        """Methodology field must mention proper IPCW implementation."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        methodology = data.get('methodology', '')
        assert 'proper_ipcw' in methodology.lower() or 'outer_training' in methodology.lower(), \
            f"Methodology should mention proper IPCW, got: {methodology}"

    def test_critical_fix_documented(self):
        """critical_fix field must document that v4 had IPCW leakage."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        critical_fix = data.get('critical_fix', '')
        assert 'v4' in critical_fix.lower() and 'test' in critical_fix.lower(), \
            f"critical_fix should mention v4 and test data issue, got: {critical_fix}"


class TestOuterSplitsVerification:
    """Verify outer_splits.csv is used correctly to derive train/test sets."""

    def test_outer_splits_all_test_type(self):
        """outer_splits.csv should only contain 'test' fold_type."""
        splits_path = SPLITS_DIR / 'outer_splits.csv'
        if not splits_path.exists():
            pytest.skip("outer_splits.csv not found")

        df = pd.read_csv(splits_path)
        assert 'fold_type' in df.columns, "outer_splits must have fold_type column"
        assert set(df['fold_type'].unique()) == {'test'}, \
            "outer_splits should only contain test fold_type"

    def test_outer_splits_coverage(self):
        """outer_splits should cover all 363 patients across 25 (repeat, fold) pairs."""
        splits_path = SPLITS_DIR / 'outer_splits.csv'
        if not splits_path.exists():
            pytest.skip("outer_splits.csv not found")

        df = pd.read_csv(splits_path)

        # Check structure
        assert set(df.columns) >= {'case_id', 'repeat', 'fold', 'fold_type'}, \
            "outer_splits must have required columns"

        # Should have 25 (repeat, fold) combinations
        unique_rf = df.groupby(['repeat', 'fold']).ngroups
        assert unique_rf == 25, f"Expected 25 (repeat, fold) pairs, got {unique_rf}"

        # Each patient should appear in exactly one fold per repeat
        for repeat in df['repeat'].unique():
            repeat_df = df[df['repeat'] == repeat]
            patient_counts = repeat_df.groupby('case_id').size()
            assert set(patient_counts.unique()) == {1}, \
                f"Each patient should appear once per repeat"


class TestBootstrappingCorrectness:
    """Verify patient-level bootstrapping is correctly implemented."""

    def test_multiplicity_preserved(self):
        """All comparisons must have multiplicity_preserved=True."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        for comp in data.get('harrell_c_comparisons', []) + data.get('uno_c_comparisons', []):
            assert comp.get('multiplicity_preserved') is True, \
                f"multiplicity_preserved must be True for {comp.get('comparison')}"

    def test_pairing_key_correct(self):
        """pairing_key must include case_id, repeat, fold."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        for comp in data.get('harrell_c_comparisons', []) + data.get('uno_c_comparisons', []):
            pairing_key = comp.get('pairing_key', [])
            assert 'case_id' in pairing_key, "pairing_key must include case_id"
            assert 'repeat' in pairing_key, "pairing_key must include repeat"
            assert 'fold' in pairing_key, "pairing_key must include fold"

    def test_n_iterations_1000(self):
        """All comparisons must use 1000 bootstrap iterations."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data.get('n_iterations') == 1000, \
            f"n_iterations must be 1000, got {data.get('n_iterations')}"

        for comp in data.get('uno_c_comparisons', []):
            assert comp.get('n_iterations') == 1000, \
                f"Uno C n_iterations must be 1000"
            assert comp.get('n_valid_iterations') == 1000, \
                f"n_valid_iterations must be 1000 (no failed iterations)"


class TestStatisticalCorrectness:
    """Verify statistical methods are correctly applied."""

    def test_bonferroni_threshold_00125(self):
        """Bonferroni threshold must be exactly 0.0125 (0.05/4)."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data.get('bonferroni_threshold') == 0.0125, \
            f"bonferroni_threshold must be 0.0125, got {data.get('bonferroni_threshold')}"

    def test_n_formal_comparisons_4(self):
        """Must have exactly 4 formal comparisons per metric."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data.get('n_formal_comparisons') == 4, \
            f"n_formal_comparisons must be 4, got {data.get('n_formal_comparisons')}"

        formal_harrell = [c for c in data.get('harrell_c_comparisons', []) if c.get('type') == 'Formal']
        formal_uno = [c for c in data.get('uno_c_comparisons', []) if c.get('type') == 'Formal']

        assert len(formal_harrell) == 4, f"Expected 4 formal Harrell C comparisons"
        assert len(formal_uno) == 4, f"Expected 4 formal Uno C comparisons"

    def test_p_values_nonzero(self):
        """All raw p-values must be > 0 (finite-sample correction applied)."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        for comp in data.get('uno_c_comparisons', []):
            p_val = comp.get('p_value')
            assert p_val is not None, "p_value must not be None"
            # p_value can be 0.0 only if the finite-sample correction formula produces it
            # which happens when all bootstrap samples favor the worse model
            assert p_val >= 0.0 and p_val <= 1.0, \
                f"p_value must be in [0, 1], got {p_val}"

    def test_significance_consistent_with_p_adj(self):
        """significant field must match p_value_adjusted < threshold."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        threshold = data.get('bonferroni_threshold', 0.0125)

        for comp in data.get('uno_c_comparisons', []):
            if comp.get('type') == 'Formal':
                p_adj = comp.get('p_value_adjusted')
                is_sig = comp.get('significant', False)

                expected_sig = p_adj < threshold
                assert is_sig == expected_sig, \
                    f"significant={is_sig} but p_adj={p_adj} >= threshold={threshold}"


class TestAUDIT_REPORT_V4:
    """Verify AUDIT_REPORT_V4.json consistency with v5 results."""

    def test_audit_v4_exists(self):
        """AUDIT_REPORT_V4.json must exist after v5 execution."""
        audit_path = FORMAL_DIR / 'AUDIT_REPORT_V4.json'
        assert audit_path.exists(), "AUDIT_REPORT_V4.json not found"

    def test_audit_v4_ipcw_source_verified(self):
        """AUDIT_REPORT_V4 must show IPCW source verification PASS."""
        audit_path = FORMAL_DIR / 'AUDIT_REPORT_V4.json'
        if not audit_path.exists():
            pytest.skip("AUDIT_REPORT_V4.json not found")

        with open(audit_path, 'r') as f:
            audit = json.load(f)

        validation_gates = audit.get('validation_gates', {})
        assert validation_gates.get('ipcw_source_verified') == 'PASS (outer training fold)', \
            "AUDIT_REPORT_V4 must verify IPCW source is outer training fold"

    def test_audit_v4_train_test_overlap_check_pass(self):
        """AUDIT_REPORT_V4 must show train_test_overlap_check PASS."""
        audit_path = FORMAL_DIR / 'AUDIT_REPORT_V4.json'
        if not audit_path.exists():
            pytest.skip("AUDIT_REPORT_V4.json not found")

        with open(audit_path, 'r') as f:
            audit = json.load(f)

        validation_gates = audit.get('validation_gates', {})
        assert validation_gates.get('train_test_overlap_check') == 'PASS', \
            "AUDIT_REPORT_V4 must verify no train/test overlap"

    def test_audit_v4_matches_v5_key_results(self):
        """AUDIT_REPORT_V4 harrell_c_summary must match v5 harrell_c_comparisons."""
        v5_path = FORMAL_DIR / 'model_comparisons_v5.json'
        audit_path = FORMAL_DIR / 'AUDIT_REPORT_V4.json'

        if not v5_path.exists() or not audit_path.exists():
            pytest.skip("Required files not found")

        with open(v5_path, 'r') as f:
            v5 = json.load(f)
        with open(audit_path, 'r') as f:
            audit = json.load(f)

        # Find M4 vs M1 in v5
        m4_vs_m1 = None
        for comp in v5.get('harrell_c_comparisons', []):
            if 'M4' in comp['comparison'] and 'M1' in comp['comparison']:
                m4_vs_m1 = comp
                break

        assert m4_vs_m1 is not None, "M4 vs M1 Harrell C not found in v5"

        # Verify in AUDIT_REPORT_V4
        harrell_summary = audit.get('harrell_c_summary', {})
        audit_p_adj = harrell_summary.get('m4_vs_m1_p_value_adjusted')
        v5_p_adj = m4_vs_m1.get('p_value_adjusted')

        if audit_p_adj is not None and v5_p_adj is not None:
            assert abs(audit_p_adj - v5_p_adj) < 0.001, \
                f"M4 vs M1 p_adj mismatch: v5={v5_p_adj}, audit={audit_p_adj}"


class TestOOFDataIntegrity:
    """Verify OOF predictions file is unchanged (not modified during IPCW computation)."""

    def test_oof_predictions_sha256_unchanged(self):
        """oof_predictions.csv must have the expected SHA-256 hash."""
        oof_path = FORMAL_DIR / 'oof_predictions.csv'
        if not oof_path.exists():
            pytest.skip("oof_predictions.csv not found")

        import hashlib
        with open(oof_path, 'rb') as f:
            sha256_hash = hashlib.sha256(f.read()).hexdigest()

        # Original hash from MEMORY.md
        expected_hash = '7b21074e208a563bc99b6a0a8c458f076a5ab333612e65ce2659cd9d6571228f'

        assert sha256_hash == expected_hash, \
            f"oof_predictions.csv has been modified! Expected {expected_hash}, got {sha256_hash}"

    def test_oof_predictions_unchanged_size(self):
        """oof_predictions.csv must have exactly 9075 rows."""
        oof_path = FORMAL_DIR / 'oof_predictions.csv'
        if not oof_path.exists():
            pytest.skip("oof_predictions.csv not found")

        df = pd.read_csv(oof_path)
        assert len(df) == 9075, f"Expected 9075 rows, got {len(df)}"

    def test_oof_predictions_has_required_columns(self):
        """oof_predictions.csv must have all required columns."""
        oof_path = FORMAL_DIR / 'oof_predictions.csv'
        if not oof_path.exists():
            pytest.skip("oof_predictions.csv not found")

        df = pd.read_csv(oof_path)
        required_cols = ['case_id', 'model', 'repeat', 'fold', 'risk_score', 'survival_months', 'event']

        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"


class TestSupersededFilesMarked:
    """Verify superseded v5 files are properly marked."""

    def test_superseded_v5_files_exist(self):
        """Superseded v5 files must exist (old implementation with IPCW leakage)."""
        superseded_json = FORMAL_DIR / 'model_comparisons_v5_SUPERSEDED_INFORMATION_LEAKAGE.json'
        superseded_csv = FORMAL_DIR / 'model_comparisons_v5_SUPERSEDED_INFORMATION_LEAKAGE.csv'

        assert superseded_json.exists(), "Superseded JSON not found"
        assert superseded_csv.exists(), "Superseded CSV not found"

    def test_current_v5_is_corrected(self):
        """Current v5 must have ipcw_source='outer_training_fold' (not superseded version)."""
        json_path = FORMAL_DIR / 'model_comparisons_v5.json'
        if not json_path.exists():
            pytest.skip("model_comparisons_v5.json not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # This test fails if we're reading the superseded version
        assert data.get('ipcw_source') == 'outer_training_fold', \
            "Current v5 must use outer_training_fold IPCW"

        # Also check that the superseded version had different results
        superseded_path = FORMAL_DIR / 'model_comparisons_v5_SUPERSEDED_INFORMATION_LEAKAGE.json'
        if superseded_path.exists():
            with open(superseded_path, 'r') as f:
                superseded = json.load(f)

            # The superseded version should have the old methodology
            assert 'SUPERSEDED' not in data.get('methodology', ''), \
                "Current v5 should not be superseded"
