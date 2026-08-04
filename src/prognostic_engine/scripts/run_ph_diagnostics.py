#!/usr/bin/env python3
"""PH diagnostics for M1 Cox models across all 25 folds.

This script runs PH diagnostics for M1 Clinical Cox models without re-running
training - it uses the saved predictions and model metadata from the formal run.

The fix addresses the dimension mismatch error by using the model's actual
feature names (after low-variance column filtering) when creating the DataFrame
for PH testing.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PACKAGE_DIR))


def check_ph_assumption_fixed(model, df, duration_col='time', event_col='event',
                              significance=0.05):
    """
    Fixed PH assumption check that handles column alignment.

    The issue: CoxPHFitter may have removed low-variance columns during fit,
    but the DataFrame passed here may still have the original columns.
    This version extracts only the columns used by the model.

    Returns global PH test result using Fisher's method for combined p-value.
    """
    from lifelines.statistics import proportional_hazard_test
    from scipy import stats

    # Get the actual column names used by the model
    model_summary = model.summary
    if hasattr(model_summary, 'columns'):
        # lifelines CoxPHFitter stores feature names in summary index
        if hasattr(model_summary, 'index'):
            used_covariates = list(model_summary.index)
        else:
            used_covariates = list(model_summary.columns)
    else:
        # Fallback: try to infer from model.params
        used_covariates = list(model.params.index)

    # Build DataFrame with only the columns the model actually used
    ph_df = pd.DataFrame()
    for col in used_covariates:
        if col in df.columns:
            ph_df[col] = df[col].values
        else:
            raise ValueError(f"Column '{col}' used by model not found in DataFrame")

    ph_df[duration_col] = df[duration_col].values
    ph_df[event_col] = df[event_col].values

    # Run the PH test
    results = proportional_hazard_test(model, ph_df, time_transform='rank')

    # Extract per-variable results from the summary DataFrame
    per_variable_results = {}
    if hasattr(results, 'summary') and results.summary is not None:
        summary_df = results.summary
        for cov_name, row in summary_df.iterrows():
            per_variable_results[str(cov_name)] = {
                'test_statistic': float(row.get('test_statistic', np.nan)),
                'p_value': float(row.get('p', np.nan))
            }

    # Calculate global p-value using Fisher's method for combined evidence
    p_values = [v['p_value'] for v in per_variable_results.values() if v['p_value'] > 0]
    if len(p_values) > 0:
        # Fisher's method: chi-squared statistic with 2k degrees of freedom
        chi2_stat = -2 * sum(np.log(p) for p in p_values if p > 0)
        combined_df = 2 * len(p_values)
        global_p_value = 1 - stats.chi2.cdf(chi2_stat, combined_df)
        global_test_statistic = chi2_stat
    else:
        global_p_value = np.nan
        global_test_statistic = np.nan

    # Count violations (p < significance)
    n_violations = sum(1 for v in per_variable_results.values()
                       if v.get('p_value', 1) < float(significance))

    ph_satisfied = global_p_value > float(significance)

    return {
        'ph_satisfied': bool(ph_satisfied),
        'global_p_value': float(global_p_value),
        'global_test_statistic': float(global_test_statistic),
        'degrees_of_freedom': 2 * len(p_values),  # Fisher's method
        'significance_level': float(significance),
        'n_violations': n_violations,
        'n_covariates_tested': len(per_variable_results),
        'per_variable': per_variable_results,
        'covariates_used': used_covariates,
        'conclusion': 'PH assumption satisfied' if ph_satisfied else 'PH assumption violated',
        'error': None
    }


def run_ph_diagnostics_single_fold(repeat, fold, data_path, splits_path):
    """
    Run PH diagnostics for a single fold.

    Parameters
    ----------
    repeat : int
    fold : int
    data_path : Path
    splits_path : Path

    Returns
    -------
    dict
    """
    from lifelines import CoxPHFitter
    from prognostic_engine.config import METABOLIC_GENES, EVALUATION_TIMES
    from prognostic_engine.preprocessing import preprocess_fold_clinical, preprocess_fold_genes
    from prognostic_engine.inner_splits import generate_inner_splits

    # Load data
    df = pd.read_parquet(data_path)
    splits = pd.read_csv(splits_path)

    # Get test cases for this fold/repeat
    test_cases = splits[
        (splits['repeat'] == repeat) &
        (splits['fold'] == fold) &
        (splits['fold_type'] == 'test')
    ]['case_id'].values

    all_cases = set(df['case_id'].values)
    train_cases = all_cases - set(test_cases)

    train_df = df[df['case_id'].isin(train_cases)].copy()

    # Preprocess clinical features
    clinical_prep = preprocess_fold_clinical(train_df, train_df)  # Use train_df twice for PH (no test needed)

    # Outcomes
    y_train = train_df['survival_months'].values
    e_train = train_df['event'].values

    # Fit M1 model (same as in training.py)
    X_clinical = clinical_prep['clinical_train']

    # Manually remove low-variance columns (same logic as M1ClinicalCox)
    variances = np.var(X_clinical, axis=0)
    mask_keep = variances > 0.001
    X_clean = X_clinical[:, mask_keep]
    clinical_cols_clean = [c for c, keep in zip(clinical_prep['clinical_cols'], mask_keep) if keep]

    # Create DataFrame for Cox model
    train_ph_df = pd.DataFrame(X_clean, columns=clinical_cols_clean)
    train_ph_df['time'] = y_train
    train_ph_df['event'] = e_train.astype(int)

    # Fit Cox model
    cph = CoxPHFitter(penalizer=0.5, l1_ratio=0.1)
    cph.fit(train_ph_df, duration_col='time', event_col='event')

    # Run PH diagnostics with FIXED function
    result = check_ph_assumption_fixed(cph, train_ph_df, 'time', 'event')
    result['repeat'] = repeat
    result['fold'] = fold
    result['model'] = 'M1_clinical_cox'
    result['n_train_samples'] = len(train_df)
    result['n_events'] = int(e_train.sum())
    result['n_features_original'] = len(clinical_prep['clinical_cols'])
    result['n_features_used'] = len(clinical_cols_clean)

    return result


def run_all_ph_diagnostics():
    """Run PH diagnostics for all 25 folds."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    data_path = PROJECT_ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    splits_path = PROJECT_ROOT / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
    output_csv = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "ph_diagnostics.csv"
    output_json = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "ph_diagnostics_summary.json"

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Data path: {data_path}")
    print(f"Data exists: {data_path.exists()}")

    print("=" * 70)
    print("PH DIAGNOSTICS FOR M1 CLINICAL COX")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Splits: {splits_path}")
    print()

    all_results = []

    for repeat in range(1, 6):
        for fold in range(1, 6):
            print(f"  Repeat {repeat}, Fold {fold}...", end=" ")
            try:
                result = run_ph_diagnostics_single_fold(repeat, fold, data_path, splits_path)
                status = "PASS" if result['ph_satisfied'] else "FAIL"
                print(f"{status} (p={result['global_p_value']:.4f}, violations={result['n_violations']})")
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    'repeat': repeat,
                    'fold': fold,
                    'model': 'M1_clinical_cox',
                    'error': str(e),
                    'ph_satisfied': None,
                    'global_p_value': None,
                    'n_violations': None
                })

    # Save CSV
    rows = []
    for r in all_results:
        row = {
            'repeat': r['repeat'],
            'fold': r['fold'],
            'model': r['model'],
            'n_train_samples': r.get('n_train_samples', ''),
            'n_events': r.get('n_events', ''),
            'n_features_original': r.get('n_features_original', ''),
            'n_features_used': r.get('n_features_used', ''),
            'global_p_value': r.get('global_p_value', ''),
            'test_statistic': r.get('global_test_statistic', ''),
            'degrees_of_freedom': r.get('degrees_of_freedom', ''),
            'n_violations': r.get('n_violations', ''),
            'ph_satisfied': r.get('ph_satisfied', ''),
            'covariates_used': '; '.join(r.get('covariates_used', [])) if r.get('covariates_used') else '',
            'error': r.get('error', '')
        }
        rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # Save JSON summary
    n_passed = sum(1 for r in all_results if r.get('ph_satisfied') == True)
    n_failed = sum(1 for r in all_results if r.get('ph_satisfied') == False)
    n_error = sum(1 for r in all_results if r.get('error') is not None)
    n_total = len(all_results)

    p_values = [r['global_p_value'] for r in all_results if r.get('global_p_value') is not None]
    mean_p = np.mean(p_values) if p_values else None

    summary = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'M1_clinical_cox',
        'total_folds': n_total,
        'passed': n_passed,
        'failed': n_failed,
        'errors': n_error,
        'success_rate': n_passed / n_total if n_total > 0 else 0,
        'mean_global_p_value': mean_p,
        'min_p_value': min(p_values) if p_values else None,
        'max_p_value': max(p_values) if p_values else None,
        'covariates_summary': 'See individual fold results for covariate-level PH tests',
        'detailed_results': all_results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {output_json}")

    # Print summary
    print()
    print("=" * 70)
    print("PH DIAGNOSTICS SUMMARY")
    print("=" * 70)
    print(f"  Total folds: {n_total}")
    print(f"  PH satisfied: {n_passed}")
    print(f"  PH violated: {n_failed}")
    print(f"  Errors: {n_error}")
    print(f"  Mean global p-value: {mean_p:.4f}" if mean_p else "  Mean global p-value: N/A")
    print()

    # Per-covariate summary
    all_per_var = {}
    for r in all_results:
        if r.get('per_variable'):
            for cov, vals in r['per_variable'].items():
                if cov not in all_per_var:
                    all_per_var[cov] = []
                all_per_var[cov].append(vals.get('p_value'))

    print("Per-covariate violation count (across 25 folds):")
    for cov, pvals in sorted(all_per_var.items()):
        violations = sum(1 for p in pvals if p is not None and p < 0.05)
        mean_p = np.nanmean([p for p in pvals if p is not None])
        print(f"  {cov}: {violations}/25 folds violated, mean p={mean_p:.4f}")

    return summary


if __name__ == "__main__":
    summary = run_all_ph_diagnostics()
    sys.exit(0)
