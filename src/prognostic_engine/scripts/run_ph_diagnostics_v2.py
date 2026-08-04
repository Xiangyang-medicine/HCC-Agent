#!/usr/bin/env python3
"""PH diagnostics v2 for M1 Cox models with proper reporting.

This is a corrected version of run_ph_diagnostics.py that:

1. Reports separate categories:
   - diagnostics_executed: total folds attempted
   - diagnostics_errors: execution failures
   - folds_with_any_covariate_violation: any covariate p < 0.05
   - folds_without_detected_violation: all covariates p >= 0.05

2. Does NOT use Fisher's method as a "global test" - this is misleading.
   Fisher's method combines per-covariate p-values, which is NOT equivalent
   to the standard global Schoenfeld test.

3. Reports per-covariate raw and Bonferroni-corrected p-values.

4. Documents that standard global test is not available for penalized Cox.

Per independent audit requirements:
- Fisher's method ≠ global Schoenfeld test
- Cannot claim "25/25 PH satisfied" based on Fisher's method
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"  # src/prognostic_engine/src
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PACKAGE_DIR))


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def check_ph_assumption_proper(
    model,
    df,
    duration_col='time',
    event_col='event',
    significance=0.05
) -> dict:
    """
    PH assumption check with proper per-covariate reporting.

    This function:
    1. Runs per-covariate PH tests
    2. Reports raw p-values
    3. Reports Bonferroni-corrected p-values
    4. Does NOT compute Fisher's method as a "global test"

    Note: The standard global Schoenfeld test is not directly available
    for penalized Cox models. Fisher's method on per-covariate p-values
    is NOT equivalent to the global test and should not be reported as such.
    """
    from lifelines.statistics import proportional_hazard_test

    # Get the actual column names used by the model
    model_summary = model.summary
    if hasattr(model_summary, 'index'):
        used_covariates = list(model_summary.index)
    else:
        used_covariates = list(model_summary.columns)

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

    # Extract per-variable results
    per_variable_results = {}
    if hasattr(results, 'summary') and results.summary is not None:
        summary_df = results.summary
        for cov_name, row in summary_df.iterrows():
            per_variable_results[str(cov_name)] = {
                'test_statistic': float(row.get('test_statistic', np.nan)),
                'p_value_raw': float(row.get('p', np.nan)),
                'p_value_bonferroni': None,  # Will compute below
                'violation_raw': False,       # Will compute below
                'violation_bonferroni': False
            }

    # Compute Bonferroni correction
    n_covariates = len(per_variable_results)
    bonferroni_alpha = significance / n_covariates if n_covariates > 0 else significance

    for cov_name, vals in per_variable_results.items():
        raw_p = vals['p_value_raw']
        vals['p_value_bonferroni'] = min(1.0, raw_p * n_covariates) if raw_p is not None else None
        vals['violation_raw'] = raw_p < significance if raw_p is not None else False
        vals['violation_bonferroni'] = vals['p_value_bonferroni'] < significance if vals['p_value_bonferroni'] is not None else False

    # Count violations
    n_violations_raw = sum(1 for v in per_variable_results.values() if v['violation_raw'])
    n_violations_bonferroni = sum(1 for v in per_variable_results.values() if v['violation_bonferroni'])

    # IMPORTANT: We do NOT compute Fisher's method here
    # Fisher's method combines per-covariate p-values, which is NOT the global test
    # The global Schoenfeld test requires running a single multivariate test

    return {
        'diagnostics_executed': 1,
        'diagnostics_errors': 0,
        'n_covariates_tested': n_covariates,
        'bonferroni_alpha': bonferroni_alpha,
        'significance_level': significance,
        'n_violations_raw': n_violations_raw,
        'n_violations_bonferroni': n_violations_bonferroni,
        'folds_with_any_covariate_violation': n_violations_raw > 0,  # At least one p < 0.05
        'folds_without_detected_violation': n_violations_raw == 0,   # All p >= 0.05
        'per_variable': per_variable_results,
        'covariates_used': used_covariates,
        # Note: We do NOT report a "global_p_value" because Fisher's method
        # is not equivalent to the standard global Schoenfeld test
        'methodology_note': (
            "Per-covariate PH tests with Bonferroni correction. "
            "Standard global Schoenfeld test not available for penalized Cox models. "
            "Fisher's method on per-covariate p-values is NOT equivalent to global test."
        ),
        'error': None
    }


def run_ph_diagnostics_single_fold(repeat: int, fold: int, data_path: Path, splits_path: Path) -> dict:
    """Run PH diagnostics for a single fold."""
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
    clinical_prep = preprocess_fold_clinical(train_df, train_df)

    # Outcomes
    y_train = train_df['survival_months'].values
    e_train = train_df['event'].values

    # Get clinical features
    X_clinical = clinical_prep['clinical_train']

    # Remove low-variance columns (same logic as M1ClinicalCox)
    variances = np.var(X_clinical, axis=0)
    mask_keep = variances > 0.001
    X_clean = X_clinical[:, mask_keep]
    clinical_cols_clean = [c for c, keep in zip(clinical_prep['clinical_cols'], mask_keep) if keep]

    # Create DataFrame for Cox model
    train_ph_df = pd.DataFrame(X_clean, columns=clinical_cols_clean)
    train_ph_df['time'] = y_train
    train_ph_df['event'] = e_train.astype(int)

    # Fit Cox model (same as in training.py)
    cph = CoxPHFitter(penalizer=0.5, l1_ratio=0.1)
    cph.fit(train_ph_df, duration_col='time', event_col='event')

    # Run PH diagnostics
    result = check_ph_assumption_proper(cph, train_ph_df, 'time', 'event')
    result['repeat'] = repeat
    result['fold'] = fold
    result['model'] = 'M1_clinical_cox'
    result['n_train_samples'] = len(train_df)
    result['n_events'] = int(e_train.sum())
    result['n_features_original'] = len(clinical_prep['clinical_cols'])
    result['n_features_used'] = len(clinical_cols_clean)

    return result


def run_all_ph_diagnostics_v2():
    """Run PH diagnostics for all 25 folds with proper reporting."""
    data_path = PROJECT_ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"
    splits_path = PROJECT_ROOT / "experiments" / "phase3a" / "splits" / "outer_splits.csv"
    output_csv = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "ph_diagnostics_v2.csv"
    output_json = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "ph_diagnostics_v2.json"

    print("=" * 70)
    print("PH DIAGNOSTICS v2 - PROPER REPORTING")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Splits: {splits_path}")
    print()

    all_results = []
    n_errors = 0

    for repeat in range(1, 6):
        for fold in range(1, 6):
            print(f"  Repeat {repeat}, Fold {fold}...", end=" ")
            try:
                result = run_ph_diagnostics_single_fold(repeat, fold, data_path, splits_path)
                status = "OK" if result['folds_without_detected_violation'] else "VIOLATION"
                print(f"{status} (violations={result['n_violations_raw']}, covs={result['n_covariates_tested']})")
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")
                n_errors += 1
                all_results.append({
                    'repeat': repeat,
                    'fold': fold,
                    'model': 'M1_clinical_cox',
                    'error': str(e),
                    'diagnostics_executed': 0,
                    'diagnostics_errors': 1,
                    'folds_with_any_covariate_violation': None,
                    'folds_without_detected_violation': None,
                    'n_violations_raw': None,
                    'n_violations_bonferroni': None,
                    'n_covariates_tested': None
                })

    # Aggregate statistics
    n_total = len(all_results)
    n_executed = sum(1 for r in all_results if r.get('diagnostics_executed', 0) > 0)
    n_errors_total = sum(1 for r in all_results if r.get('error') is not None)
    n_with_violation = sum(1 for r in all_results if r.get('folds_with_any_covariate_violation') == True)
    n_without_violation = sum(1 for r in all_results if r.get('folds_without_detected_violation') == True)

    # Per-covariate summary across all folds
    all_per_var = {}
    for r in all_results:
        if r.get('per_variable'):
            for cov, vals in r['per_variable'].items():
                if cov not in all_per_var:
                    all_per_var[cov] = {'raw_pvals': [], 'bonf_pvals': [], 'violations_raw': [], 'violations_bonf': []}
                all_per_var[cov]['raw_pvals'].append(vals.get('p_value_raw'))
                all_per_var[cov]['bonf_pvals'].append(vals.get('p_value_bonferroni'))
                all_per_var[cov]['violations_raw'].append(vals.get('violation_raw', False))
                all_per_var[cov]['violations_bonf'].append(vals.get('violation_bonferroni', False))

    covariate_summary = {}
    for cov, data in sorted(all_per_var.items()):
        raw_pvals = [p for p in data['raw_pvals'] if p is not None]
        bonf_pvals = [p for p in data['bonf_pvals'] if p is not None]
        covariate_summary[cov] = {
            'violations_raw': sum(data['violations_raw']),
            'violations_bonferroni': sum(data['violations_bonf']),
            'mean_pvalue_raw': float(np.nanmean(raw_pvals)) if raw_pvals else None,
            'mean_pvalue_bonferroni': float(np.nanmean(bonf_pvals)) if bonf_pvals else None,
            'total_folds': len(raw_pvals)
        }

    # Save CSV
    rows = []
    for r in all_results:
        row = {
            'repeat': r.get('repeat', ''),
            'fold': r.get('fold', ''),
            'model': r.get('model', 'M1_clinical_cox'),
            'n_train_samples': r.get('n_train_samples', ''),
            'n_events': r.get('n_events', ''),
            'n_covariates_tested': r.get('n_covariates_tested', ''),
            'bonferroni_alpha': r.get('bonferroni_alpha', ''),
            'n_violations_raw': r.get('n_violations_raw', ''),
            'n_violations_bonferroni': r.get('n_violations_bonferroni', ''),
            'folds_with_any_covariate_violation': r.get('folds_with_any_covariate_violation', ''),
            'folds_without_detected_violation': r.get('folds_without_detected_violation', ''),
            'error': r.get('error', '')
        }
        rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # Save JSON summary
    summary = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'M1_clinical_cox',
        'methodology': 'Per-covariate PH tests with Bonferroni correction',
        'methodology_warning': (
            "Standard global Schoenfeld test not available for penalized Cox models. "
            "Fisher's method on per-covariate p-values is NOT equivalent to global test."
        ),
        # Proper reporting categories
        'diagnostics_executed': n_executed,
        'diagnostics_errors': n_errors_total,
        'folds_with_any_covariate_violation': n_with_violation,
        'folds_without_detected_violation': n_without_violation,
        # Note: We do NOT report "passed" or "failed" counts
        # because Fisher's method was incorrectly used as "global test"
        'total_folds': n_total,
        'covariate_summary': covariate_summary,
        'detailed_results': all_results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=_json_serializer)
    print(f"Saved: {output_json}")

    # Print summary
    print()
    print("=" * 70)
    print("PH DIAGNOSTICS v2 SUMMARY")
    print("=" * 70)
    print()
    print("IMPORTANT NOTE:")
    print("  Fisher's method is NOT equivalent to the global Schoenfeld test.")
    print("  The standard global test is not available for penalized Cox models.")
    print()
    print("Reporting Categories:")
    print(f"  diagnostics_executed:        {n_executed}")
    print(f"  diagnostics_errors:          {n_errors_total}")
    print(f"  folds_with_any_violation:    {n_with_violation}")
    print(f"  folds_without_violation:     {n_without_violation}")
    print()
    print("Per-Covariate Violations (Bonferroni-corrected, across 25 folds):")
    for cov, data in sorted(covariate_summary.items()):
        sig_note = " **SIGNIFICANT**" if data['violations_bonferroni'] > 0 else ""
        print(f"  {cov}: {data['violations_raw']}/25 raw violations, "
              f"{data['violations_bonferroni']}/25 Bonferroni violations"
              f"{sig_note}")
    print()

    # Verify minimum p > 0
    all_raw_pvals = [p for cov_data in all_per_var.values() for p in cov_data['raw_pvals'] if p is not None]
    if all_raw_pvals:
        min_p = min(all_raw_pvals)
        print(f"Minimum raw p-value: {min_p:.6f}")
        if min_p > 0:
            print("All p-values > 0 (no numerical issues)")
        else:
            print("WARNING: Some p-values = 0")

    return summary


if __name__ == "__main__":
    summary = run_all_ph_diagnostics_v2()
    sys.exit(0)
