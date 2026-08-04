"""Main nested CV training pipeline for Phase 3A.

Implements the locked nested-CV protocol per SAP v1.1:
- 5 outer repeats × 5 outer folds
- 5 inner folds for hyperparameter tuning
- Proper one-hot encoding
- Complete metrics (Harrell C, Uno C, AUC, Brier, IBS, calibration)
- Bootstrap comparison
- PH diagnostics
- Integrity gates

CORRECTIONS per Phase 3A methodological reset:
1. Fixed: compute_all_metrics now requires train data for IPCW estimation
2. Added: Integrity gates at each model checkpoint
3. Added: Proper status tracking per SAP v1.1
"""

import json
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from prognostic_engine.config import (
    METABOLIC_GENES, N_OUTER_REPEATS, N_OUTER_FOLDS, N_INNER_FOLDS,
    OUTER_SEED, INNER_SEED, EVALUATION_TIMES, OUTPUT_DIR
)
from prognostic_engine.inner_splits import (
    generate_inner_splits, save_inner_splits, extract_inner_fold_data
)
from prognostic_engine.inner_preprocessing import (
    InnerFoldPreprocessor, preprocess_inner_fold_clinical, preprocess_inner_fold_genes
)
from prognostic_engine.preprocessing import preprocess_fold_clinical, preprocess_fold_genes
from prognostic_engine.models import M1ClinicalCox, M2M3Coxnet, M4RSF, M5DeepSurv
from prognostic_engine.metrics import (
    compute_all_metrics, harrell_c_index, uno_c_index
)
from prognostic_engine.bootstrap import patient_level_paired_bootstrap
from prognostic_engine.ph_diagnostics import check_ph_assumption


# Suppress pycox warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)


EXPECTED_MODELS = (
    'M1_clinical_cox',
    'M2_gene_elasticnet',
    'M3_combined_elasticnet',
    'M4_combined_rsf',
    'M5_deepsurv',
)
EXPECTED_PATIENTS = 363
FULL_OUTER_REPEATS = 5

# Sensitivity Analysis Patient Counts
SA_CONFIG = {
    'SA1': {'n_patients': 363, 'predictions_per_model': 1815, 'total': 9075},
    'SA2': {'n_patients': 361, 'predictions_per_model': 1805, 'total': 9025},
    'SA3': {'n_patients': 338, 'predictions_per_model': 1690, 'total': 8450},
}


def _json_default(value):
    """Serialize scientific Python values without stringifying booleans."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, Path)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# Integrity Monitoring (non-blocking) per SAP v1.1 Phase 3A Reset
# These are for diagnostics only - do NOT block training based on performance
INTEGRITY_MONITORING = {
    'c_index_min': 0.50,       # Diagnostic threshold (not a gate)
    'c_index_max': 0.99,       # Diagnostic threshold (possible overfitting)
    'auc_min': 0.50,           # Diagnostic threshold
    'ibs_max': 0.50,           # Diagnostic threshold
    'event_rate_min': 0.10,    # Minimum for reliable estimates
    'na_ratio_max': 0.20,      # Max NaN predictions
}


class IntegrityMonitor:
    """Diagnostic integrity monitor for model predictions per SAP v1.1.

    Per Phase 3A reset: This is for diagnostics only - does NOT block training.
    """

    def __init__(self, model_name, repeat, fold):
        self.model_name = model_name
        self.repeat = repeat
        self.fold = fold
        self.checks = {}
        self.warnings = []

    def check_c_index(self, value, metric_name='c_index'):
        """Check C-index is within reasonable bounds (diagnostic only)."""
        if np.isnan(value):
            self.checks[metric_name] = {'status': 'INFO', 'reason': 'NaN value (estimation failed)'}
            self.warnings.append(f'{metric_name}: NaN (NOT_ESTIMABLE)')
        elif value < INTEGRITY_MONITORING['c_index_min']:
            self.checks[metric_name] = {'status': 'WARN',
                                        'value': value,
                                        'reason': f'{value:.3f} < {INTEGRITY_MONITORING["c_index_min"]} (below random)'}
            self.warnings.append(f'{metric_name}: {value:.3f} (below random)')
        elif value > INTEGRITY_MONITORING['c_index_max']:
            self.checks[metric_name] = {'status': 'WARN',
                                        'value': value,
                                        'reason': f'{value:.3f} > {INTEGRITY_MONITORING["c_index_max"]} (possible overfitting)'}
            self.warnings.append(f'{metric_name}: {value:.3f} (possible overfitting)')
        else:
            self.checks[metric_name] = {'status': 'OK', 'value': value}

    def check_auc(self, auc_dict):
        """Check time-dependent AUC values (diagnostic only)."""
        for key, value in auc_dict.items():
            if np.isnan(value):
                self.checks[key] = {'status': 'INFO', 'reason': 'NaN (NOT_ESTIMABLE)'}
                self.warnings.append(f'{key}: NaN (NOT_ESTIMABLE)')
            elif value < INTEGRITY_MONITORING['auc_min']:
                self.checks[key] = {'status': 'WARN',
                                    'value': value,
                                    'reason': f'{value:.3f} < {INTEGRITY_MONITORING["auc_min"]}'}
                self.warnings.append(f'{key}: {value:.3f} (below random)')

    def check_ibs(self, value):
        """Check Integrated Brier Score (diagnostic only)."""
        if np.isnan(value):
            self.warnings.append('IBS: NaN (NOT_ESTIMABLE)')
        elif value > INTEGRITY_MONITORING['ibs_max']:
            self.warnings.append(f'IBS: {value:.3f} (poor calibration)')

    def check_event_rate(self, events, n_samples):
        """Check event rate in test set (diagnostic only)."""
        rate = events.sum() / n_samples
        if rate < INTEGRITY_MONITORING['event_rate_min']:
            self.warnings.append(f'Event rate: {rate:.1%} (may affect reliability)')

    def check_predictions(self, risk_scores, survival_probs):
        """Check prediction quality (diagnostic only)."""
        nan_ratio = (np.isnan(risk_scores).sum() + np.isnan(survival_probs).sum()) / (len(risk_scores) + survival_probs.size)
        if nan_ratio > INTEGRITY_MONITORING['na_ratio_max']:
            self.warnings.append(f'NaN ratio: {nan_ratio:.1%} (high)')
        elif nan_ratio > 0:
            self.warnings.append(f'NaN ratio: {nan_ratio:.1%}')

    def get_status(self):
        """Get monitoring status (always returns PASS for diagnostics)."""
        return {
            'model': self.model_name,
            'repeat': self.repeat,
            'fold': self.fold,
            'status': 'MONITORED',  # Always monitored, never blocked
            'warnings': self.warnings,
            'checks': self.checks
        }


def run_integrity_monitor(model_name, repeat, fold, metrics, risk_scores, survival_probs, y_test, e_test):
    """Run integrity monitoring (non-blocking diagnostics only).

    Per Phase 3A Reset: This is for diagnostics only - does NOT block training.
    """
    monitor = IntegrityMonitor(model_name, repeat, fold)

    # Check C-indices
    if 'harrell_c' in metrics:
        monitor.check_c_index(metrics['harrell_c'], 'harrell_c')
    if 'uno_c' in metrics:
        monitor.check_c_index(metrics['uno_c'], 'uno_c')

    # Check AUC
    auc_keys = [k for k in metrics.keys() if k.startswith('auc_') and not k.endswith('_status')]
    if auc_keys:
        monitor.check_auc({k: metrics[k] for k in auc_keys})

    # Check IBS
    if 'ibs' in metrics:
        monitor.check_ibs(metrics['ibs'])

    # Check event rate
    monitor.check_event_rate(e_test, len(y_test))

    # Check predictions
    monitor.check_predictions(risk_scores, survival_probs)

    status = monitor.get_status()

    # Print monitoring info (never blocks)
    if status['warnings']:
        print(f" [warnings: {', '.join(status['warnings'])}]", end="")
    else:
        print(" [metrics OK]", end="")

    return status


class NestedCVTrainer:
    """Formal nested CV trainer for Phase 3A survival models."""

    def __init__(self, data_path, splits_path, output_dir=None, sa_name=None):
        """
        Initialize trainer.

        Parameters
        ----------
        data_path : str/Path
            Path to modeling dataset parquet file
        splits_path : str/Path
            Path to outer_splits.csv
        output_dir : str/Path, optional
            Output directory for results
        sa_name : str, optional
            Sensitivity analysis name ('SA2', 'SA3'). If None, uses formal SA1 config.
        """
        self.data_path = Path(data_path)
        self.splits_path = Path(splits_path)
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sa_name = sa_name

        self.df = None
        self.splits = None
        self.all_predictions = []
        self.metrics_by_model = defaultdict(list)
        self.ph_results = defaultdict(list)
        self.model_failures = defaultdict(list)  # Per Phase 3A reset: track failures with traceback

    def load_data(self):
        """Load data and splits."""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        self.df = pd.read_parquet(self.data_path)
        self.splits = pd.read_csv(self.splits_path)

        print(f"  Dataset: {len(self.df)} patients, {self.df['event'].sum()} events")
        print(f"  Events: {self.df['event'].mean()*100:.1f}%")
        print(f"  Outer splits: {len(self.splits)} test entries")

    def _preprocess_inner_fold(self, inner_splits, train_df, feature_type='genes'):
        """
        Perform inner-fold preprocessing for hyperparameter tuning.

        Per Phase 3A reset: Each inner fold must:
        1. Extract inner_train_df and inner_val_df by case_id
        2. Fit preprocessing ONLY on inner_train_df
        3. Transform inner_val_df only

        Parameters
        ----------
        inner_splits : dict
            Inner CV splits with case_id lists
        train_df : pd.DataFrame
            Full outer training DataFrame
        feature_type : str
            'genes', 'clinical', or 'combined'

        Returns
        -------
        list of dicts
            Each dict contains preprocessed X_train, X_val, y_train, y_val, e_train, e_val
        """
        from prognostic_engine.config import METABOLIC_GENES

        gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]
        preprocessor = InnerFoldPreprocessor()
        results = []

        for fold_info in inner_splits['folds']:
            train_case_ids = fold_info['train_case_ids']
            val_case_ids = fold_info['val_case_ids']

            # Extract DataFrames by case_id (NOT by row index)
            inner_train_df, inner_val_df = extract_inner_fold_data(train_df, train_case_ids, val_case_ids)

            if feature_type == 'clinical':
                X_train_dict = preprocess_inner_fold_clinical(inner_train_df, inner_val_df)
                X_train = X_train_dict['clinical_train']
                X_val = X_train_dict['clinical_val']
            elif feature_type == 'genes':
                X_train_dict = preprocess_inner_fold_genes(inner_train_df, inner_val_df, gene_cols)
                X_train = X_train_dict['train_genes']
                X_val = X_train_dict['val_genes']
            else:  # combined
                clinical_prep = preprocess_inner_fold_clinical(inner_train_df, inner_val_df)
                gene_prep = preprocess_inner_fold_genes(inner_train_df, inner_val_df, gene_cols)
                X_train = np.hstack([clinical_prep['clinical_train'], gene_prep['train_genes']])
                X_val = np.hstack([clinical_prep['clinical_val'], gene_prep['val_genes']])

            y_train = inner_train_df['survival_months'].values
            y_val = inner_val_df['survival_months'].values
            e_train = inner_train_df['event'].values
            e_val = inner_val_df['event'].values

            results.append({
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'e_train': e_train,
                'e_val': e_val,
                'train_case_ids': train_case_ids,
                'val_case_ids': val_case_ids,
            })

        return results

    def run_training_fold(self, repeat, fold):
        """
        Run training for a single outer fold with inner CV tuning.

        Per Phase 3A reset: Inner-fold preprocessing isolation is implemented:
        - M1 (no tuning): Uses full outer-train preprocessing (acceptable - no inner CV)
        - M2, M3, M4, M5: Preprocessing is fit ONLY on inner-training data per fold

        Parameters
        ----------
        repeat : int
            Outer repeat number (1-indexed)
        fold : int
            Outer fold number (1-indexed)

        Returns
        -------
        dict
            Results for this fold
        """
        from prognostic_engine.config import METABOLIC_GENES
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored

        # Get test cases for this fold/repeat
        test_cases = self.splits[
            (self.splits['repeat'] == repeat) &
            (self.splits['fold'] == fold) &
            (self.splits['fold_type'] == 'test')
        ]['case_id'].values

        all_cases = set(self.df['case_id'].values)
        train_cases = all_cases - set(test_cases)

        train_df = self.df[self.df['case_id'].isin(train_cases)].copy()
        test_df = self.df[self.df['case_id'].isin(test_cases)].copy()

        print(f"\n    Outer: Repeat {repeat}, Fold {fold}")
        print(f"    Train: {len(train_df)}, Test: {len(test_df)}")

        # Per Phase 3A reset: Generate inner splits FIRST with case_id lists
        train_case_ids_list = list(train_cases)
        inner_splits = generate_inner_splits(train_case_ids_list, repeat, fold)

        # Save inner splits
        save_inner_splits(inner_splits, self.output_dir / "inner_splits")

        # For M1: Use full outer-train preprocessing (M1 doesn't tune, so no inner CV)
        # Per Phase 3A reset: This is acceptable for M1 since there's no inner fold
        clinical_prep = preprocess_fold_clinical(train_df, test_df)
        gene_prep = preprocess_fold_genes(train_df, test_df)

        # Build feature matrices for M1
        X_clinical_train = clinical_prep['clinical_train']
        X_clinical_test = clinical_prep['clinical_test']
        X_gene_train = gene_prep['train_genes']
        X_gene_test = gene_prep['test_genes']  # Fixed: was 'val_genes'
        combined_cols = clinical_prep['clinical_cols'] + gene_prep['gene_cols']
        X_combined_train = np.hstack([X_clinical_train, X_gene_train])
        X_combined_test = np.hstack([X_clinical_test, X_gene_test])

        # Outcomes
        y_train = train_df['survival_months'].values
        y_test = test_df['survival_months'].values
        e_train = train_df['event'].values
        e_test = test_df['event'].values
        test_case_ids = test_df['case_id'].values

        results = {}

        # M1: Clinical-only Cox (no tuning) - uses full outer-train preprocessing
        print(f"      M1: Clinical Cox PH...")
        try:
            m1 = M1ClinicalCox()
            m1.fit(X_clinical_train, y_train, e_train, clinical_prep['clinical_cols'])
            risk_test = m1.predict_risk(X_clinical_test)
            survival_test = m1.predict_survival(X_clinical_test, EVALUATION_TIMES)

            # Compute metrics - PASS TRAIN DATA for IPCW estimation
            m1_metrics = compute_all_metrics(
                y_train, e_train,              # Train data for IPCW
                y_test, e_test,                # Test data for evaluation
                risk_test, survival_test,       # Predictions
                times=EVALUATION_TIMES
            )
            results['M1'] = m1_metrics
            self.metrics_by_model['M1_clinical_cox'].append(m1_metrics)

            # Integrity monitoring (diagnostic only, never blocks)
            ig_status = run_integrity_monitor('M1_clinical_cox', repeat, fold,
                                          m1_metrics, risk_test, survival_test, y_test, e_test)
            if not hasattr(self, 'integrity_results'):
                self.integrity_results = []
            self.integrity_results.append(ig_status)

            # Store predictions
            self._store_predictions(test_case_ids, repeat, fold, 'M1_clinical_cox',
                                  risk_test, survival_test, y_test, e_test)

            # PH diagnostics for M1 (skip if fails)
            try:
                ph_train_df = pd.DataFrame(X_clinical_train, columns=clinical_prep['clinical_cols'])
                ph_train_df['time'] = y_train
                ph_train_df['event'] = e_train
                m1_ph = check_ph_assumption(m1.model, ph_train_df, 'time', 'event')
                self.ph_results['M1'].append(m1_ph)
            except Exception as ph_err:
                self.ph_results['M1'].append({'error': str(ph_err)})

            print(f"        Harrell C: {m1_metrics['harrell_c']:.3f}, Uno C: {m1_metrics['uno_c']:.3f}")
        except Exception as e:
            print(f"        M1 ERROR: {e}")
            traceback.print_exc()
            results['M1'] = {'harrell_c': np.nan, 'uno_c': np.nan}
            # Per Phase 3A reset: Track failures with traceback
            self.model_failures['M1_clinical_cox'].append({
                'repeat': repeat,
                'fold': fold,
                'error': str(e),
                'traceback': traceback.format_exc()
            })

        # M2: Gene-only Coxnet (with TRUE inner-fold preprocessing isolation)
        # Per Phase 3A reset: preprocessing must be fit ONLY on inner_train_df
        print(f"      M2: Gene Coxnet (tuning with inner-fold preprocessing)...")
        try:
            m2 = M2M3Coxnet('M2')

            # Per Phase 3A reset: Use dynamic alpha path from model.alphas_
            # First, fit on first fold to get alpha path, then use those alphas
            gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]

            # Get alpha path from initial fit
            first_train_df, first_val_df = extract_inner_fold_data(
                train_df,
                inner_splits['folds'][0]['train_case_ids'],
                inner_splits['folds'][0]['val_case_ids']
            )
            gene_prep_first = preprocess_inner_fold_genes(first_train_df, first_val_df, gene_cols)
            X_first = gene_prep_first['train_genes']
            y_first = first_train_df['survival_months'].values
            e_first = first_train_df['event'].values
            y_first_struct = np.array(
                [(bool(e), t) for e, t in zip(e_first, y_first)],
                dtype=[('event', bool), ('time', float)]
            )

            # Get dynamic alpha path
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            alpha_fitter = CoxnetSurvivalAnalysis(l1_ratio=0.5, max_iter=100000)
            alpha_fitter.fit(X_first, y_first_struct)
            dynamic_alphas = alpha_fitter.alphas_[::max(1, len(alpha_fitter.alphas_) // 20)][:10]  # Sample ~10 alphas

            l1_ratio_range = [0.1, 0.3, 0.5, 0.7, 0.9]
            best_score = -np.inf
            best_params = (dynamic_alphas[0] if len(dynamic_alphas) > 0 else 0.1, 0.5)

            for alpha in dynamic_alphas:
                for l1_ratio in l1_ratio_range:
                    if l1_ratio <= 0 or l1_ratio > 1:
                        continue

                    scores = []
                    for fold_info in inner_splits['folds']:
                        train_case_ids = fold_info['train_case_ids']
                        val_case_ids = fold_info['val_case_ids']

                        # Per Phase 3A reset: extract by case_id, fit on inner_train ONLY
                        inner_train_df, inner_val_df = extract_inner_fold_data(
                            train_df, train_case_ids, val_case_ids
                        )

                        # Fit preprocessing on inner_train_df
                        gene_prep_inner = preprocess_inner_fold_genes(
                            inner_train_df, inner_val_df, gene_cols
                        )
                        X_t = gene_prep_inner['train_genes']
                        X_v = gene_prep_inner['val_genes']

                        y_t = inner_train_df['survival_months'].values
                        y_v = inner_val_df['survival_months'].values
                        e_t = inner_train_df['event'].values
                        e_v = inner_val_df['event'].values

                        # Structured array for sksurv
                        y_t_struct = np.array(
                            [(bool(e), t) for e, t in zip(e_t, y_t)],
                            dtype=[('event', bool), ('time', float)]
                        )
                        y_v_struct = np.array(
                            [(bool(e), t) for e, t in zip(e_v, y_v)],
                            dtype=[('event', bool), ('time', float)]
                        )

                        try:
                            model = CoxnetSurvivalAnalysis(
                                alphas=[alpha], l1_ratio=l1_ratio, max_iter=100000
                            )
                            model.fit(X_t, y_t_struct)
                        except ArithmeticError:
                            continue

                        risk_pred = model.predict(X_v)

                        # Check for zero-coefficient model
                        coef = model.coef_
                        if coef is None or np.abs(coef).sum() < 1e-6:
                            continue

                        try:
                            from sksurv.metrics import concordance_index_censored
                            cidx, _, _, _, _ = concordance_index_censored(
                                y_v_struct['event'], y_v_struct['time'], risk_pred
                            )
                            if not np.isnan(cidx):
                                scores.append(cidx)
                        except Exception:
                            pass

                    if scores:
                        mean_score = np.mean(scores)
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = (alpha, l1_ratio)

            best_alpha, best_l1 = best_params
            print(f"        Tuned: alpha={best_alpha:.6f}, l1_ratio={best_l1:.2f}")

            # Step 2: Fit preprocessing on FULL outer_train for final model
            gene_prep_full = preprocess_inner_fold_genes(train_df, test_df, gene_cols)
            X_gene_train_inner = gene_prep_full['train_genes']
            X_gene_test_inner = gene_prep_full['val_genes']

            m2.fit(X_gene_train_inner, y_train, e_train, gene_cols,
                   alpha=best_alpha, l1_ratio=best_l1)
            risk_test = m2.predict_risk(X_gene_test_inner)
            survival_test = m2.predict_survival(X_gene_test_inner, EVALUATION_TIMES)

            # Compute metrics - PASS TRAIN DATA for IPCW estimation
            m2_metrics = compute_all_metrics(
                y_train, e_train,              # Train data for IPCW
                y_test, e_test,                # Test data for evaluation
                risk_test, survival_test,       # Predictions
                times=EVALUATION_TIMES
            )
            results['M2'] = m2_metrics
            self.metrics_by_model['M2_gene_elasticnet'].append(m2_metrics)

            # Integrity monitoring (diagnostic only, never blocks)
            ig_status = run_integrity_monitor('M2_gene_elasticnet', repeat, fold,
                                          m2_metrics, risk_test, survival_test, y_test, e_test)
            self.integrity_results.append(ig_status)

            self._store_predictions(test_case_ids, repeat, fold, 'M2_gene_elasticnet',
                                  risk_test, survival_test, y_test, e_test)

            print(f"        Harrell C: {m2_metrics['harrell_c']:.3f}, Uno C: {m2_metrics['uno_c']:.3f}")
        except Exception as e:
            print(f"        M2 ERROR: {e}")
            traceback.print_exc()
            results['M2'] = {'harrell_c': np.nan, 'uno_c': np.nan}
            self.model_failures['M2_gene_elasticnet'].append({
                'repeat': repeat, 'fold': fold, 'error': str(e),
                'traceback': traceback.format_exc()
            })

        # M3: Combined Coxnet (with TRUE inner-fold preprocessing isolation)
        # Per Phase 3A reset: preprocessing must be fit ONLY on inner_train_df
        print(f"      M3: Combined Coxnet (tuning with inner-fold preprocessing)...")
        try:
            m3 = M2M3Coxnet('M3')

            # Per Phase 3A reset: Use dynamic alpha path from model.alphas_
            gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]

            # Get alpha path from initial fit
            first_train_df, first_val_df = extract_inner_fold_data(
                train_df,
                inner_splits['folds'][0]['train_case_ids'],
                inner_splits['folds'][0]['val_case_ids']
            )
            clinical_prep_first = preprocess_inner_fold_clinical(first_train_df, first_val_df)
            gene_prep_first = preprocess_inner_fold_genes(first_train_df, first_val_df, gene_cols)
            X_first = np.hstack([clinical_prep_first['clinical_train'], gene_prep_first['train_genes']])
            y_first = first_train_df['survival_months'].values
            e_first = first_train_df['event'].values
            y_first_struct = np.array(
                [(bool(e), t) for e, t in zip(e_first, y_first)],
                dtype=[('event', bool), ('time', float)]
            )

            # Get dynamic alpha path
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            alpha_fitter = CoxnetSurvivalAnalysis(l1_ratio=0.5, max_iter=100000)
            alpha_fitter.fit(X_first, y_first_struct)
            dynamic_alphas = alpha_fitter.alphas_[::max(1, len(alpha_fitter.alphas_) // 20)][:10]  # Sample ~10 alphas

            l1_ratio_range = [0.1, 0.3, 0.5, 0.7, 0.9]
            best_score = -np.inf
            best_params = (dynamic_alphas[0] if len(dynamic_alphas) > 0 else 0.1, 0.5)

            for alpha in dynamic_alphas:
                for l1_ratio in l1_ratio_range:
                    if l1_ratio <= 0 or l1_ratio > 1:
                        continue

                    scores = []
                    for fold_info in inner_splits['folds']:
                        train_case_ids = fold_info['train_case_ids']
                        val_case_ids = fold_info['val_case_ids']

                        # Per Phase 3A reset: extract by case_id, fit on inner_train ONLY
                        inner_train_df, inner_val_df = extract_inner_fold_data(
                            train_df, train_case_ids, val_case_ids
                        )

                        # Fit preprocessing on inner_train_df
                        clinical_prep_inner = preprocess_inner_fold_clinical(inner_train_df, inner_val_df)
                        gene_prep_inner = preprocess_inner_fold_genes(inner_train_df, inner_val_df, gene_cols)
                        X_t = np.hstack([clinical_prep_inner['clinical_train'], gene_prep_inner['train_genes']])
                        X_v = np.hstack([clinical_prep_inner['clinical_val'], gene_prep_inner['val_genes']])

                        y_t = inner_train_df['survival_months'].values
                        y_v = inner_val_df['survival_months'].values
                        e_t = inner_train_df['event'].values
                        e_v = inner_val_df['event'].values

                        # Structured array for sksurv
                        y_t_struct = np.array(
                            [(bool(e), t) for e, t in zip(e_t, y_t)],
                            dtype=[('event', bool), ('time', float)]
                        )
                        y_v_struct = np.array(
                            [(bool(e), t) for e, t in zip(e_v, y_v)],
                            dtype=[('event', bool), ('time', float)]
                        )

                        try:
                            model = CoxnetSurvivalAnalysis(
                                alphas=[alpha], l1_ratio=l1_ratio, max_iter=100000
                            )
                            model.fit(X_t, y_t_struct)
                        except ArithmeticError:
                            continue

                        risk_pred = model.predict(X_v)

                        # Check for zero-coefficient model
                        coef = model.coef_
                        if coef is None or np.abs(coef).sum() < 1e-6:
                            continue

                        try:
                            from sksurv.metrics import concordance_index_censored
                            cidx, _, _, _, _ = concordance_index_censored(
                                y_v_struct['event'], y_v_struct['time'], risk_pred
                            )
                            if not np.isnan(cidx):
                                scores.append(cidx)
                        except Exception:
                            pass

                    if scores:
                        mean_score = np.mean(scores)
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = (alpha, l1_ratio)

            best_alpha, best_l1 = best_params
            print(f"        Tuned: alpha={best_alpha:.6f}, l1_ratio={best_l1:.2f}")

            # Step 2: Fit preprocessing on FULL outer_train for final model
            clinical_prep_full = preprocess_inner_fold_clinical(train_df, test_df)
            gene_prep_full = preprocess_inner_fold_genes(train_df, test_df, gene_cols)
            X_combined_train_inner = np.hstack([
                clinical_prep_full['clinical_train'], gene_prep_full['train_genes']
            ])
            X_combined_test_inner = np.hstack([
                clinical_prep_full['clinical_val'], gene_prep_full['val_genes']
            ])
            combined_cols = clinical_prep_full['clinical_cols'] + gene_cols

            m3.fit(X_combined_train_inner, y_train, e_train, combined_cols,
                   alpha=best_alpha, l1_ratio=best_l1)
            risk_test = m3.predict_risk(X_combined_test_inner)
            survival_test = m3.predict_survival(X_combined_test_inner, EVALUATION_TIMES)

            # Compute metrics - PASS TRAIN DATA for IPCW estimation
            m3_metrics = compute_all_metrics(
                y_train, e_train,              # Train data for IPCW
                y_test, e_test,                # Test data for evaluation
                risk_test, survival_test,       # Predictions
                times=EVALUATION_TIMES
            )
            results['M3'] = m3_metrics
            self.metrics_by_model['M3_combined_elasticnet'].append(m3_metrics)

            # Integrity monitoring (diagnostic only, never blocks)
            ig_status = run_integrity_monitor('M3_combined_elasticnet', repeat, fold,
                                          m3_metrics, risk_test, survival_test, y_test, e_test)
            self.integrity_results.append(ig_status)

            self._store_predictions(test_case_ids, repeat, fold, 'M3_combined_elasticnet',
                                  risk_test, survival_test, y_test, e_test)

            print(f"        Harrell C: {m3_metrics['harrell_c']:.3f}, Uno C: {m3_metrics['uno_c']:.3f}")
        except Exception as e:
            print(f"        M3 ERROR: {e}")
            traceback.print_exc()
            results['M3'] = {'harrell_c': np.nan, 'uno_c': np.nan}
            self.model_failures['M3_combined_elasticnet'].append({
                'repeat': repeat, 'fold': fold, 'error': str(e),
                'traceback': traceback.format_exc()
            })

        # M4: Combined RSF (with TRUE inner-fold preprocessing isolation)
        # Per Phase 3A reset: preprocessing must be fit ONLY on inner_train_df
        print(f"      M4: Combined RSF (tuning with inner-fold preprocessing)...")
        try:
            from prognostic_engine.models import M4RSF
            from sksurv.metrics import concordance_index_censored

            m4 = M4RSF()
            gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]

            # RSF hyperparameter grid
            rsf_configs = [
                {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
                {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
                {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 3},
            ]

            best_score = -np.inf
            best_config = rsf_configs[0]

            # Inner CV for hyperparameter selection
            for config in rsf_configs:
                scores = []
                for fold_info in inner_splits['folds']:
                    train_case_ids = fold_info['train_case_ids']
                    val_case_ids = fold_info['val_case_ids']

                    # Per Phase 3A reset: extract by case_id, fit on inner_train ONLY
                    inner_train_df, inner_val_df = extract_inner_fold_data(
                        train_df, train_case_ids, val_case_ids
                    )

                    # Fit preprocessing on inner_train_df only
                    clinical_prep_inner = preprocess_inner_fold_clinical(inner_train_df, inner_val_df)
                    gene_prep_inner = preprocess_inner_fold_genes(inner_train_df, inner_val_df, gene_cols)
                    X_t = np.hstack([clinical_prep_inner['clinical_train'], gene_prep_inner['train_genes']])
                    X_v = np.hstack([clinical_prep_inner['clinical_val'], gene_prep_inner['val_genes']])

                    y_t = inner_train_df['survival_months'].values
                    y_v = inner_val_df['survival_months'].values
                    e_t = inner_train_df['event'].values
                    e_v = inner_val_df['event'].values

                    # Structured array for sksurv
                    y_t_struct = np.array(
                        [(bool(e), t) for e, t in zip(e_t, y_t)],
                        dtype=[('event', bool), ('time', float)]
                    )
                    y_v_struct = np.array(
                        [(bool(e), t) for e, t in zip(e_v, y_v)],
                        dtype=[('event', bool), ('time', float)]
                    )

                    try:
                        from sksurv.ensemble import RandomSurvivalForest
                        model = RandomSurvivalForest(**config, random_state=42, n_jobs=-1)
                        model.fit(X_t, y_t_struct)
                        risk_pred = model.predict(X_v)

                        cidx, _, _, _, _ = concordance_index_censored(
                            y_v_struct['event'], y_v_struct['time'], risk_pred
                        )
                        if not np.isnan(cidx):
                            scores.append(cidx)
                        else:
                            scores.append(0.5)
                    except Exception:
                        scores.append(0.5)

                mean_score = np.mean(scores) if scores else 0.5
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config

            print(f"        Tuned: n_estimators={best_config['n_estimators']}, max_depth={best_config['max_depth']}")

            # Step 2: Fit preprocessing on FULL outer_train for final model
            clinical_prep_full = preprocess_inner_fold_clinical(train_df, test_df)
            gene_prep_full = preprocess_inner_fold_genes(train_df, test_df, gene_cols)
            X_combined_train = np.hstack([
                clinical_prep_full['clinical_train'], gene_prep_full['train_genes']
            ])
            X_combined_test = np.hstack([
                clinical_prep_full['clinical_val'], gene_prep_full['val_genes']
            ])

            # Fit final RSF model
            y_train_struct = np.array(
                [(bool(e), t) for e, t in zip(e_train, y_train)],
                dtype=[('event', bool), ('time', float)]
            )
            from sksurv.ensemble import RandomSurvivalForest
            final_model = RandomSurvivalForest(**best_config, random_state=42, n_jobs=-1)
            final_model.fit(X_combined_train, y_train_struct)

            risk_test = final_model.predict(X_combined_test)
            survival_test = final_model.predict_survival_function(X_combined_test)

            # Convert survival functions to probability matrix
            n_patients = len(X_combined_test)
            survival_probs = np.zeros((n_patients, len(EVALUATION_TIMES)))
            for j in range(n_patients):
                sf = survival_test[j]
                for i, t in enumerate(EVALUATION_TIMES):
                    survival_probs[j, i] = sf(t)

            # Compute metrics - PASS TRAIN DATA for IPCW estimation
            m4_metrics = compute_all_metrics(
                y_train, e_train,              # Train data for IPCW
                y_test, e_test,                # Test data for evaluation
                risk_test, survival_probs,     # Predictions
                times=EVALUATION_TIMES
            )
            results['M4'] = m4_metrics
            self.metrics_by_model['M4_combined_rsf'].append(m4_metrics)

            # Integrity monitoring (diagnostic only, never blocks)
            ig_status = run_integrity_monitor('M4_combined_rsf', repeat, fold,
                                          m4_metrics, risk_test, survival_probs, y_test, e_test)
            self.integrity_results.append(ig_status)

            self._store_predictions(test_case_ids, repeat, fold, 'M4_combined_rsf',
                                  risk_test, survival_probs, y_test, e_test)

            print(f"        Harrell C: {m4_metrics['harrell_c']:.3f}, Uno C: {m4_metrics['uno_c']:.3f}")
        except Exception as e:
            print(f"        M4 ERROR: {e}")
            traceback.print_exc()
            results['M4'] = {'harrell_c': np.nan, 'uno_c': np.nan}
            self.model_failures['M4_combined_rsf'].append({
                'repeat': repeat, 'fold': fold, 'error': str(e),
                'traceback': traceback.format_exc()
            })

        # M5: Combined DeepSurv (with TRUE inner-fold preprocessing isolation)
        # Per Phase 3A reset: preprocessing must be fit ONLY on inner_train_df
        print(f"      M5: Combined DeepSurv (tuning with inner-fold preprocessing)...")
        try:
            from prognostic_engine.models import M5DeepSurv
            from sksurv.metrics import concordance_index_censored

            m5 = M5DeepSurv()
            gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]

            # DeepSurv hyperparameter grid
            deepsurv_configs = [
                {'hidden_layers': [32], 'lr': 0.01, 'batch_frac': 1.0},
                {'hidden_layers': [64], 'lr': 0.01, 'batch_frac': 1.0},
                {'hidden_layers': [32, 16], 'lr': 0.001, 'batch_frac': 1.0},
            ]

            best_score = -np.inf
            best_config = deepsurv_configs[0]

            # Inner CV for hyperparameter selection
            for config in deepsurv_configs:
                scores = []
                for fold_info in inner_splits['folds']:
                    train_case_ids = fold_info['train_case_ids']
                    val_case_ids = fold_info['val_case_ids']

                    # Per Phase 3A reset: extract by case_id, fit on inner_train ONLY
                    inner_train_df, inner_val_df = extract_inner_fold_data(
                        train_df, train_case_ids, val_case_ids
                    )

                    # Fit preprocessing on inner_train_df only
                    clinical_prep_inner = preprocess_inner_fold_clinical(inner_train_df, inner_val_df)
                    gene_prep_inner = preprocess_inner_fold_genes(inner_train_df, inner_val_df, gene_cols)
                    X_t = np.hstack([clinical_prep_inner['clinical_train'], gene_prep_inner['train_genes']])
                    X_v = np.hstack([clinical_prep_inner['clinical_val'], gene_prep_inner['val_genes']])

                    y_t = inner_train_df['survival_months'].values
                    y_v = inner_val_df['survival_months'].values
                    e_t = inner_train_df['event'].values
                    e_v = inner_val_df['event'].values

                    try:
                        model, _ = m5._train_deepsurv(
                            X_t, y_t, e_t,
                            hidden_layers=config['hidden_layers'],
                            lr=config['lr'],
                            batch_frac=config['batch_frac'],
                            epochs=50,
                            verbose=False
                        )
                        # Convert to float32 for PyTorch compatibility
                        X_v_f32 = X_v.astype(np.float32)
                        risk_pred = model.predict(X_v_f32).flatten()

                        y_v_struct = np.array(
                            [(bool(e), t) for e, t in zip(e_v, y_v)],
                            dtype=[('event', bool), ('time', float)]
                        )
                        cidx, _, _, _, _ = concordance_index_censored(
                            y_v_struct['event'], y_v_struct['time'], risk_pred
                        )
                        if not np.isnan(cidx):
                            scores.append(cidx)
                        else:
                            scores.append(0.5)
                    except Exception:
                        scores.append(0.5)

                mean_score = np.mean(scores) if scores else 0.5
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config

            print(f"        Tuned: layers={best_config['hidden_layers']}, lr={best_config['lr']}")

            # Step 2: Fit preprocessing on FULL outer_train for final model
            clinical_prep_full = preprocess_inner_fold_clinical(train_df, test_df)
            gene_prep_full = preprocess_inner_fold_genes(train_df, test_df, gene_cols)
            X_combined_train = np.hstack([
                clinical_prep_full['clinical_train'], gene_prep_full['train_genes']
            ])
            X_combined_test = np.hstack([
                clinical_prep_full['clinical_val'], gene_prep_full['val_genes']
            ])

            # Fit final DeepSurv model
            final_model, _ = m5._train_deepsurv(
                X_combined_train, y_train, e_train,
                hidden_layers=best_config['hidden_layers'],
                lr=best_config['lr'],
                batch_frac=best_config['batch_frac'],
                epochs=100,
                verbose=True
            )

            # CRITICAL: Assign model to self.model (required by predict_survival)
            m5.model = final_model

            # Convert to float32 for PyTorch compatibility
            X_combined_test_f32 = X_combined_test.astype(np.float32)
            risk_test = final_model.predict(X_combined_test_f32).flatten()

            # Get survival probabilities from DeepSurv (uses m5.model which is now set)
            survival_probs = m5.predict_survival(X_combined_test_f32, EVALUATION_TIMES)

            # Compute metrics - PASS TRAIN DATA for IPCW estimation
            m5_metrics = compute_all_metrics(
                y_train, e_train,              # Train data for IPCW
                y_test, e_test,                # Test data for evaluation
                risk_test, survival_probs,     # Predictions
                times=EVALUATION_TIMES
            )
            results['M5'] = m5_metrics
            self.metrics_by_model['M5_deepsurv'].append(m5_metrics)

            # Integrity monitoring (diagnostic only, never blocks)
            ig_status = run_integrity_monitor('M5_deepsurv', repeat, fold,
                                          m5_metrics, risk_test, survival_probs, y_test, e_test)
            self.integrity_results.append(ig_status)

            self._store_predictions(test_case_ids, repeat, fold, 'M5_deepsurv',
                                  risk_test, survival_probs, y_test, e_test)

            print(f"        Harrell C: {m5_metrics['harrell_c']:.3f}, Uno C: {m5_metrics['uno_c']:.3f}")
        except Exception as e:
            print(f"        M5 ERROR: {e}")
            traceback.print_exc()
            results['M5'] = {'harrell_c': np.nan, 'uno_c': np.nan}
            self.model_failures['M5_deepsurv'].append({
                'repeat': repeat, 'fold': fold, 'error': str(e),
                'traceback': traceback.format_exc()
            })

    def _store_predictions(self, case_ids, repeat, fold, model_name,
                         risk_scores, survival_probs, y_time, y_event):
        """Store predictions for later aggregation."""
        for i, case_id in enumerate(case_ids):
            self.all_predictions.append({
                'case_id': case_id,
                'repeat': repeat,
                'fold': fold,
                'model': model_name,
                'risk_score': float(risk_scores[i]),
                'survival_probability_12m': float(survival_probs[i, 0]) if len(survival_probs.shape) > 1 else float(survival_probs[i]),
                'survival_probability_36m': float(survival_probs[i, 1]) if len(survival_probs.shape) > 1 else np.nan,
                'survival_probability_60m': float(survival_probs[i, 2]) if len(survival_probs.shape) > 1 else np.nan,
                'survival_months': float(y_time[i]),
                'event': int(y_event[i])
            })

    def run(self, n_repeats=None):
        """Run nested CV training.

        Parameters
        ----------
        n_repeats : int, optional
            Number of outer repeats to run. Default is 5 (full training).
            For pilot mode, use n_repeats=1.
        """
        if n_repeats is None:
            n_repeats = N_OUTER_REPEATS

        if n_repeats not in (1, FULL_OUTER_REPEATS):
            raise ValueError(
                "Only n_repeats=1 (pilot) or n_repeats=5 (locked formal protocol) is allowed"
            )

        is_pilot = n_repeats == 1
        mode_label = "PILOT" if is_pilot else "FORMAL"

        print("\n" + "=" * 70)
        print(f"PHASE 3A {mode_label} NESTED CV TRAINING")
        print("=" * 70)
        print(f"\nProtocol: {n_repeats} repeats × {N_OUTER_FOLDS} folds × {N_INNER_FOLDS} inner folds")
        print(f"Models: M1, M2, M3, M4, M5")
        print(f"Output: {self.output_dir}")

        start_time = datetime.now()

        # Load data
        self.load_data()

        # Run training
        fold_results = []
        total_folds = n_repeats * N_OUTER_FOLDS

        print("\n" + "-" * 70)
        print("TRAINING PROGRESS")
        print("-" * 70)

        for repeat in range(1, n_repeats + 1):
            for fold in range(1, N_OUTER_FOLDS + 1):
                fold_num = (repeat - 1) * N_OUTER_FOLDS + fold
                print(f"\n[{fold_num}/{total_folds}]", end="")

                try:
                    results = self.run_training_fold(repeat, fold)
                    fold_results.append({
                        'repeat': repeat,
                        'fold': fold,
                        'results': results
                    })
                except Exception as e:
                    print(f"\n  FOLD ERROR: {e}")
                    traceback.print_exc()

        # Save results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._save_results(
            fold_results,
            start_time,
            end_time,
            duration,
            expected_repeats=n_repeats,
        )

        return fold_results

    def _save_results(self, fold_results, start_time, end_time, duration, expected_repeats):
        """Save results and apply exact structural gates for pilot or formal CV."""
        if expected_repeats not in (1, FULL_OUTER_REPEATS):
            raise ValueError("expected_repeats must be 1 (pilot) or 5 (formal)")

        is_pilot = expected_repeats == 1

        # =====================================================================
        # HARDCODED PROTOCOL CONSTANTS (Phase 3A Reset)
        # =====================================================================
        # Use SA-aware expected values if SA config is available
        sa_key = self.sa_name if self.sa_name else 'SA1'
        sa_cfg = SA_CONFIG.get(sa_key, SA_CONFIG['SA1'])

        expected_folds = expected_repeats * N_OUTER_FOLDS
        expected_patients = sa_cfg['n_patients']
        expected_per_model = expected_patients * expected_repeats
        expected_total = expected_per_model * len(EXPECTED_MODELS)
        expected_fold_pairs = {
            (repeat, fold)
            for repeat in range(1, expected_repeats + 1)
            for fold in range(1, N_OUTER_FOLDS + 1)
        }

        pred_df, pred_path, csv_roundtrip_ok = self._write_predictions()

        # Aggregate metrics
        summary = self._aggregate_metrics()

        # Bootstrap comparison
        print("\nRunning bootstrap comparison...")
        comparison = self._bootstrap_comparison(pred_df)
        bootstrap_comparison_complete = bool(
            len(comparison) == 3
            and all('error' not in result for result in comparison.values())
        )

        # Aggregate integrity monitoring results (diagnostic only, never blocks training)
        integrity_summary = self._aggregate_integrity()

        # =====================================================================
        # VALIDATION: Hardcoded protocol checks
        # =====================================================================
        fold_count = len(fold_results)
        total_predictions = len(self.all_predictions)
        # For SA analyses, derive cohort from splits file (not full dataset)
        # This allows validation against SA-specific patient lists
        if self.sa_name:
            cohort_case_ids = set(self.splits['case_id'].astype(str))
        else:
            cohort_case_ids = set(self.df['case_id'].astype(str))
        cohort_size_ok = len(cohort_case_ids) == expected_patients

        # Per-model counts
        observed_models = set(pred_df['model']) if 'model' in pred_df.columns else set()
        per_model_counts = {
            model: int((pred_df['model'] == model).sum()) if 'model' in pred_df.columns else 0
            for model in EXPECTED_MODELS
        }
        extra_models = sorted(observed_models - set(EXPECTED_MODELS))

        # Check for skipped/failed models
        skipped_models = [m for m in EXPECTED_MODELS if per_model_counts[m] == 0]
        model_failures = {
            model: failures
            for model, failures in dict(self.model_failures).items()
            if failures
        }
        failed_models = sorted(model_failures)

        # Check (model, repeat, case_id) uniqueness
        key_columns = ['model', 'repeat', 'case_id']
        required_prediction_columns = key_columns + ['fold', 'risk_score']
        missing_prediction_columns = [
            column for column in required_prediction_columns if column not in pred_df.columns
        ]
        duplicate_count = (
            int(pred_df.duplicated(key_columns, keep=False).sum())
            if not missing_prediction_columns else total_predictions
        )

        # Check exact fold and patient coverage for every model/repeat.
        repeat_coverage = {}
        repeat_coverage_ok = True
        fold_coverage = {}
        fold_coverage_ok = True
        for model in EXPECTED_MODELS:
            model_fold_pairs = set(
                pred_df.loc[pred_df['model'] == model, ['repeat', 'fold']]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ) if not missing_prediction_columns else set()
            fold_coverage[model] = sorted([list(pair) for pair in model_fold_pairs])
            if model_fold_pairs != expected_fold_pairs:
                fold_coverage_ok = False

            for repeat in range(1, expected_repeats + 1):
                model_repeat_preds = pred_df[
                    (pred_df['model'] == model) & (pred_df['repeat'] == repeat)
                ]
                n_patients = model_repeat_preds['case_id'].nunique()
                actual_case_ids = set(model_repeat_preds['case_id'].astype(str))
                repeat_coverage[f"{model}_r{repeat}"] = int(n_patients)
                if n_patients != expected_patients or actual_case_ids != cohort_case_ids:
                    repeat_coverage_ok = False

        # Check for NaN/Inf in predictions
        prediction_value_columns = [
            'risk_score',
            'survival_probability_12m',
            'survival_probability_36m',
            'survival_probability_60m',
        ]
        prediction_nonfinite = {}
        for column in prediction_value_columns:
            if column not in pred_df.columns:
                prediction_nonfinite[column] = total_predictions or 1
                continue
            values = pd.to_numeric(pred_df[column], errors='coerce').to_numpy(dtype=float)
            prediction_nonfinite[column] = int((~np.isfinite(values)).sum())

        # Check for NaN/Inf in metrics
        nan_in_metrics = {}
        nonfinite_metric_details = {}
        metrics_coverage = {
            model: len(self.metrics_by_model.get(model, []))
            for model in EXPECTED_MODELS
        }
        primary_metric_keys = (
            'harrell_c', 'uno_c', 'auc_12m', 'auc_36m', 'auc_60m',
            'brier_12m', 'brier_36m', 'brier_60m', 'ibs',
        )
        for m, vals in self.metrics_by_model.items():
            if vals:
                nan_count = 0
                details = []
                for metric_index, v in enumerate(vals):
                    if v is None:
                        nan_count += 1
                        details.append({'fold_index': metric_index, 'metric': 'all'})
                    else:
                        for metric_key in primary_metric_keys:
                            metric_value = v.get(metric_key)
                            try:
                                if metric_value is None or not np.isfinite(float(metric_value)):
                                    nan_count += 1
                                    details.append({
                                        'fold_index': metric_index,
                                        'metric': metric_key,
                                    })
                            except (TypeError, ValueError):
                                nan_count += 1
                                details.append({
                                    'fold_index': metric_index,
                                    'metric': metric_key,
                                })
                nan_in_metrics[m] = nan_count
                nonfinite_metric_details[m] = details

        # Build validation dict with NATIVE BOOLEANS (not strings)
        validation = {
            "mode": "PILOT" if is_pilot else "FORMAL",
            "sensitivity_analysis": self.sa_name if self.sa_name else "SA1",
            "expected_models": list(EXPECTED_MODELS),
            "completed_models": [m for m in EXPECTED_MODELS if per_model_counts[m] > 0],
            "extra_models": extra_models,
            "expected_repeats": expected_repeats,
            "expected_folds": expected_folds,
            "completed_folds": fold_count,
            "expected_per_model": expected_per_model,
            "expected_total": expected_total,
            "actual_total": total_predictions,
            "per_model_counts": per_model_counts,
            "repeat_coverage": repeat_coverage,
            "fold_coverage": fold_coverage,
            "skipped_models": skipped_models,
            "failed_models": failed_models,
            "model_failures": model_failures,
            "missing_prediction_columns": missing_prediction_columns,
            "nonfinite_predictions": prediction_nonfinite,
            "nan_in_metrics": nan_in_metrics,
            "nonfinite_metric_details": nonfinite_metric_details,
            "metrics_coverage": metrics_coverage,
            "duplicate_records": duplicate_count,
            "cohort_size_ok": bool(cohort_size_ok),
            "all_models_completed": bool(observed_models == set(EXPECTED_MODELS)),
            "folds_complete": bool(fold_count == expected_folds),
            "fold_coverage_complete": bool(fold_coverage_ok),
            "repeat_coverage_complete": bool(repeat_coverage_ok),
            "per_model_counts_equal": bool(all(count == expected_per_model for count in per_model_counts.values())),
            "total_predictions_equal": bool(total_predictions == expected_total),
            "no_nan_in_predictions": bool(all(count == 0 for count in prediction_nonfinite.values())),
            "no_nan_in_metrics": bool(all(nc == 0 for nc in nan_in_metrics.values())),
            "metrics_coverage_complete": bool(
                all(count == expected_folds for count in metrics_coverage.values())
            ),
            "no_duplicate_records": bool(duplicate_count == 0),
            "model_failures_empty": bool(len(model_failures) == 0),
            "skipped_models_empty": bool(len(skipped_models) == 0),
            "no_extra_models": bool(len(extra_models) == 0),
            "csv_roundtrip_ok": bool(csv_roundtrip_ok),
            "bootstrap_comparison_complete": bootstrap_comparison_complete,
        }

        required_gates = [
            "cohort_size_ok", "all_models_completed", "folds_complete",
            "fold_coverage_complete", "repeat_coverage_complete",
            "per_model_counts_equal", "total_predictions_equal",
            "no_nan_in_predictions", "no_nan_in_metrics", "metrics_coverage_complete",
            "no_duplicate_records", "model_failures_empty",
            "skipped_models_empty", "no_extra_models", "csv_roundtrip_ok",
            "bootstrap_comparison_complete",
        ]
        validation["all_required_gates_passed"] = bool(
            not missing_prediction_columns and all(validation[key] for key in required_gates)
        )

        # =====================================================================
        # DETERMINE STATUS
        # =====================================================================
        if validation["all_required_gates_passed"]:
            overall_status = 'PILOT_COMPLETED' if is_pilot else 'COMPLETED'
            status_reason = None
        else:
            overall_status = 'FAILED_INCOMPLETE'
            failed_gates = [key for key in required_gates if not validation[key]]
            if missing_prediction_columns:
                failed_gates.append('required_prediction_columns_present')
            status_reason = f"Failed structural gates: {failed_gates}"

        # Print validation summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  Expected models: {len(EXPECTED_MODELS)}")
        print(f"  Completed models: {len(validation['completed_models'])}")
        print(f"  Expected folds: {expected_folds}")
        print(f"  Completed folds: {validation['completed_folds']}")
        print(f"  Expected total predictions: {expected_total}")
        print(f"  Actual total predictions: {total_predictions}")
        print(f"  Skipped models: {skipped_models if skipped_models else 'None'}")
        print(f"  Failed models: {failed_models if failed_models else 'None'}")
        print(f"  Non-finite predictions: {prediction_nonfinite}")
        print(f"  NaN/Inf in metrics: {nan_in_metrics}")
        print(f"  Status: {overall_status}")
        if status_reason:
            print(f"  Reason: {status_reason}")
        print("=" * 70)

        # Save metrics
        metrics_path = self.output_dir / "metrics_summary.json"
        report = {
            'protocol': 'PILOT_NESTED_CV' if is_pilot else 'FORMAL_NESTED_CV',
            'status': overall_status,
            'status_reason': status_reason,  # Actual reason for status
            'monitoring_type': 'NON_BLOCKING',  # Integrity monitoring never blocks training
            'training': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'duration_hours': duration / 3600
            },
            'cv_config': {
                'n_repeats': expected_repeats,
                'n_outer_folds': N_OUTER_FOLDS,
                'n_inner_folds': N_INNER_FOLDS,
                'outer_seed': OUTER_SEED,
                'inner_seed': INNER_SEED,
                'total_outer_folds': expected_folds
            },
            'models': {
                'M1_clinical_cox': 'Clinical Cox PH (no tuning)',
                'M2_gene_elasticnet': 'Gene Coxnet (inner CV tuning)',
                'M3_combined_elasticnet': 'Combined Coxnet (inner CV tuning)',
                'M4_combined_rsf': 'Combined RSF (inner CV tuning)',
                'M5_deepsurv': 'Combined DeepSurv (inner CV tuning)'
            },
            'integrity_gates': integrity_summary,
            'validation': validation,  # Hardcoded protocol validation with native booleans
            'metrics': summary,
            'bootstrap_comparison': comparison,
            'ph_diagnostics': dict(self.ph_results),
            'model_failures': model_failures  # Tracked per fold/repeat
        }

        with open(metrics_path, 'w') as f:
            json.dump(report, f, indent=2, default=_json_default)
        print(f"Saved metrics: {metrics_path}")

        # Print summary
        self._print_summary(summary, comparison, duration, overall_status, status_reason)
        return report

    def _write_predictions(self):
        """Persist OOF predictions through the canonical round-trip path."""
        pred_df = pd.DataFrame(self.all_predictions)
        pred_path = self.output_dir / "oof_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"\nSaved predictions: {pred_path}")
        roundtrip_df = pd.read_csv(pred_path)
        roundtrip_ok = bool(
            len(roundtrip_df) == len(pred_df)
            and list(roundtrip_df.columns) == list(pred_df.columns)
        )
        return pred_df, pred_path, roundtrip_ok

    def _aggregate_integrity(self):
        """Aggregate integrity monitoring results (diagnostic only)."""
        if not hasattr(self, 'integrity_results') or not self.integrity_results:
            return {'status': 'NO_MONITORING_RUN'}

        total = len(self.integrity_results)
        # All monitoring results are informational only
        with_warnings = sum(1 for g in self.integrity_results if g['warnings'])

        # Group by model
        by_model = {}
        for g in self.integrity_results:
            model = g['model']
            if model not in by_model:
                by_model[model] = {'total': 0, 'warnings': []}
            by_model[model]['total'] += 1
            by_model[model]['warnings'].extend(g['warnings'])

        return {
            'type': 'MONITORING_ONLY',  # Never blocks training
            'total_checks': total,
            'with_warnings': with_warnings,
            'warning_rate': with_warnings / total if total > 0 else 0,
            'by_model': by_model
        }

    def _aggregate_metrics(self):
        """Aggregate metrics across folds."""
        summary = {}

        for model, metrics_list in self.metrics_by_model.items():
            if not metrics_list:
                continue

            # Collect each metric type
            metric_keys = metrics_list[0].keys()
            summary[model] = {}

            for key in metric_keys:
                # Skip non-scalar metrics (like auc_results dict) - they're handled separately
                raw_values = [m[key] for m in metrics_list if key in m]
                if not raw_values:
                    continue

                # Only aggregate scalar numeric values
                scalar_values = []
                for v in raw_values:
                    if isinstance(v, (int, float)) and not (np.isnan(v) if isinstance(v, float) else False):
                        scalar_values.append(v)

                if scalar_values:
                    summary[model][key] = {
                        'n_folds': len(scalar_values),
                        'mean': np.mean(scalar_values),
                        'std': np.std(scalar_values),
                        'min': np.min(scalar_values),
                        'max': np.max(scalar_values),
                        'median': np.median(scalar_values),
                        'q25': np.percentile(scalar_values, 25),
                        'q75': np.percentile(scalar_values, 75),
                        'per_fold': scalar_values
                    }

        return summary

    def _bootstrap_comparison(self, pred_df):
        """Run patient-level paired bootstrap comparison."""
        results = {}

        model_pairs = [
            ('M2_gene_elasticnet', 'M1_clinical_cox'),
            ('M3_combined_elasticnet', 'M1_clinical_cox'),
            ('M3_combined_elasticnet', 'M2_gene_elasticnet'),
        ]

        for model_a, model_b in model_pairs:
            try:
                comparison = patient_level_paired_bootstrap(
                    pred_df,
                    n_iterations=1000,
                    seed=456,
                    comparison_pair=(model_a, model_b)
                )
                results[f'{model_a}_vs_{model_b}'] = {
                    'methodology': comparison['methodology'],
                    'iterations_requested': comparison['iterations_requested'],
                    'iterations_valid': comparison['iterations_valid'],
                    'n_patients': comparison['n_patients'],
                    'n_repeats': comparison['n_repeats'],
                    'observed_repeat_differences': comparison['observed_repeat_differences'],
                    'observed_mean_difference': comparison['observed_mean_difference'],
                    'mean_diff': comparison['mean_diff'],
                    'ci_lower': comparison['ci_lower'],
                    'ci_upper': comparison['ci_upper'],
                    'p_value': comparison['p_value'],
                    'significant': comparison['p_value'] < 0.05
                }
            except Exception as e:
                results[f'{model_a}_vs_{model_b}'] = {'error': str(e)}

        return results

    def _print_summary(self, summary, comparison, duration, status='COMPLETED', status_reason=None):
        """Print summary table."""
        print("\n" + "=" * 80)
        print("FORMAL NESTED CV RESULTS")
        print("=" * 80)

        print(f"\n{'Model':<30} {'N':>5} {'Harrell C':>12} {'Uno C':>10} {'AUC 36m':>10} {'IBS':>8}")
        print("-" * 80)

        for model in ['M1_clinical_cox', 'M2_gene_elasticnet', 'M3_combined_elasticnet',
                      'M4_combined_rsf', 'M5_deepsurv']:
            if model not in summary:
                print(f"{model:<30} {'--':>5} {'--':>12} {'--':>10} {'--':>10} {'--':>8}")
                continue

            m = summary[model]
            n = m.get('harrell_c', {}).get('n_folds', '--')
            hc = m.get('harrell_c', {}).get('mean', '--')
            uc = m.get('uno_c', {}).get('mean', '--')
            auc = m.get('auc_36m', {}).get('mean', '--')
            ibs = m.get('ibs', {}).get('mean', '--')

            hc_str = f"{hc:.3f}" if isinstance(hc, (int, float)) and not np.isnan(hc) else "--"
            uc_str = f"{uc:.3f}" if isinstance(uc, (int, float)) and not np.isnan(uc) else "--"
            auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) and not np.isnan(auc) else "--"
            ibs_str = f"{ibs:.3f}" if isinstance(ibs, (int, float)) and not np.isnan(ibs) else "--"

            print(f"{model:<30} {n:>5} {hc_str:>12} {uc_str:>10} {auc_str:>10} {ibs_str:>8}")

        print(f"\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Status: {status}")
        if status_reason:
            print(f"Reason: {status_reason}")


def main():
    """Main entry point."""
    from pathlib import Path

    # Paths - training.py is in src/prognostic_engine/src/prognostic_engine/
    # So parent.parent.parent = src/prognostic_engine, parent.parent.parent.parent = src
    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    splits_path = project_root / "experiments" / "phase3a" / "splits" / "outer_splits.csv"

    trainer = NestedCVTrainer(data_path, splits_path)
    trainer.run()


if __name__ == "__main__":
    main()
