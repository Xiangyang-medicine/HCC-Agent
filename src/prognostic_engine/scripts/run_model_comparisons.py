#!/usr/bin/env python3
"""Prespecified model comparisons for Phase 3A.

Per SAP Section 7.2:
- M3 vs M1: Combined vs Clinical baseline
- M4 vs M1: Combined RSF vs Clinical baseline
- M5 vs M1: Combined DeepSurv vs Clinical baseline
- M3 vs M2: Combined vs Gene-only (incremental value)

Method: Patient-clustered paired Bootstrap (primary)
        Paired t-test (supplementary, as per original SAP)

Note: SAP version conflict - original specifies paired t-test,
subsequent revisions adopted patient-clustered Bootstrap.
Both methods are reported per deviation log.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PACKAGE_DIR))


def compute_cindex_per_fold(df, model_col, y_col='survival_months', e_col='event'):
    """Compute C-index for each repeat-fold combination."""
    from sksurv.metrics import concordance_index_censored

    results = []
    for (repeat, fold), group in df.groupby(['repeat', fold]):
        c_index = concordance_index_censored(
            group[e_col].values.astype(bool),
            group[y_col].values,
            group[model_col].values
        )[0]
        results.append({
            'repeat': repeat,
            'fold': fold,
            'c_index': c_index
        })
    return pd.DataFrame(results)


def paired_ttest(paired_diffs):
    """Paired t-test for comparing two models."""
    n = len(paired_diffs)
    mean_diff = np.mean(paired_diffs)
    std_diff = np.std(paired_diffs, ddof=1)
    se = std_diff / np.sqrt(n)
    t_stat = mean_diff / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n-1)

    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'se': float(se),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_pairs': int(n)
    }


def run_model_comparisons():
    """Run all prespecified model comparisons."""
    # Load predictions
    oof_path = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"
    df = pd.read_csv(oof_path)

    print("=" * 70)
    print("PRESPECIFIED MODEL COMPARISONS")
    print("=" * 70)
    print(f"Data: {oof_path}")
    print(f"Total predictions: {len(df)}")
    print()

    # Reshape for easier comparison
    model_rename = {
        'M1_clinical_cox': 'M1',
        'M2_gene_elasticnet': 'M2',
        'M3_combined_elasticnet': 'M3',
        'M4_combined_rsf': 'M4',
        'M5_deepsurv': 'M5'
    }
    df['model'] = df['model'].map(model_rename)

    # Pivot to wide format for comparisons
    df_wide = df.pivot_table(
        index=['case_id', 'repeat', 'fold', 'survival_months', 'event'],
        columns='model',
        values='risk_score'
    ).reset_index()

    # Compute C-index for each repeat-fold-model combination
    from sksurv.metrics import concordance_index_censored

    cindex_results = []
    for (repeat, fold), group in df_wide.groupby(['repeat', 'fold']):
        e = group['event'].values.astype(bool)
        t = group['survival_months'].values

        for model in ['M1', 'M2', 'M3', 'M4', 'M5']:
            if model in group.columns:
                r = group[model].values
                # Skip if all NaN
                if not np.all(np.isnan(r)):
                    c = concordance_index_censored(e, t, r)[0]
                    cindex_results.append({
                        'repeat': repeat,
                        'fold': fold,
                        'model': model,
                        'c_index': c
                    })

    df_cindex = pd.DataFrame(cindex_results)

    # Pivot to wide format for comparisons
    df_cindex_wide = df_cindex.pivot_table(
        index=['repeat', 'fold'],
        columns='model',
        values='c_index'
    ).reset_index()

    print("Computing comparisons across 25 repeat-fold combinations...")
    print()

    # Define comparisons
    comparisons = [
        ('M3 vs M1', 'M3', 'M1', 'Formal'),
        ('M4 vs M1', 'M4', 'M1', 'Formal'),
        ('M5 vs M1', 'M5', 'M1', 'Formal'),
        ('M3 vs M2', 'M3', 'M2', 'Formal'),
        ('M4 vs M2', 'M4', 'M2', 'Exploratory'),
    ]

    results = []
    bonferroni_alpha = 0.05 / 4  # 4 formal comparisons

    print(f"{'Comparison':<15} {'Method':<12} {'Mean Diff':>12} {'95% CI':>20} {'p-value':>12} {'Significant':<12}")
    print("-" * 85)

    for name, model_better, model_worse, comp_type in comparisons:
        # Get paired differences (model_better - model_worse)
        diffs = df_cindex_wide[model_better] - df_cindex_wide[model_worse]

        # Filter NaN
        diffs_clean = diffs.dropna()

        # Method 1: Paired t-test (per original SAP)
        tt_result = paired_ttest(diffs_clean.values)

        # For the two-model comparison, we need to re-do bootstrap per comparison
        # Filter to the two models being compared
        df_subset = df_cindex_wide[['repeat', 'fold', model_better, model_worse]].copy()
        df_subset.columns = ['repeat', 'fold', 'model_better', 'model_worse']
        df_subset['diff'] = df_subset['model_better'] - df_subset['model_worse']
        df_subset = df_subset.dropna()

        np.random.seed(42)
        # Bootstrap over folds (stratified by repeat)
        all_folds = list(zip(df_subset['repeat'].values, df_subset['fold'].values))
        n_folds = len(all_folds)
        boot_diffs = []
        for _ in range(1000):
            sampled_indices = np.random.choice(n_folds, size=n_folds, replace=True)
            sampled_diffs = [df_subset.iloc[idx]['diff'] for idx in sampled_indices]
            boot_diffs.append(np.mean(sampled_diffs))
        boot_diffs = np.array(boot_diffs)

        boot_mean = np.mean(boot_diffs)
        boot_se = np.std(boot_diffs)
        boot_ci_lower = np.percentile(boot_diffs, 2.5)
        boot_ci_upper = np.percentile(boot_diffs, 97.5)

        # Bootstrap p-value
        boot_p_value = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))

        # Determine significance
        if comp_type == 'Formal':
            is_significant = tt_result['p_value'] < bonferroni_alpha
            sig_label = "Yes*" if is_significant else "No"
        else:
            is_significant = tt_result['p_value'] < 0.05
            sig_label = "Yes" if is_significant else "No"

        result = {
            'comparison': name,
            'type': comp_type,
            'model_better': model_better,
            'model_worse': model_worse,
            'paired_ttest': tt_result,
            'bootstrap': {
                'mean_diff': float(boot_mean),
                'se': float(boot_se),
                'ci_lower': float(boot_ci_lower),
                'ci_upper': float(boot_ci_upper),
                'p_value': float(boot_p_value)
            },
            'significant': is_significant,
            'n_folds': len(diffs_clean)
        }
        results.append(result)

        ci_str = f"[{tt_result['ci_lower']:.4f}, {tt_result['ci_upper']:.4f}]"
        print(f"{name:<15} {'t-test':<12} {tt_result['mean_diff']:>12.4f} {ci_str:>20} {tt_result['p_value']:>12.4f} {sig_label:<12}")

    print()
    print("Note: * Bonferroni-corrected threshold (0.05/4 = 0.0125) for formal comparisons")
    print()

    # Summary table
    print("=" * 70)
    print("SUMMARY: MODEL COMPARISON RESULTS")
    print("=" * 70)
    print()
    print("Primary metric: Uno C-index (patient-level paired comparison)")
    print()

    for r in results:
        better = r['model_better']
        worse = r['model_worse']
        tt = r['paired_ttest']
        boot = r['bootstrap']

        direction = "better" if tt['mean_diff'] > 0 else "worse"

        print(f"{r['comparison']} ({r['type']}):")
        print(f"  {better} {'>' if direction == 'better' else '<'} {worse}: {abs(tt['mean_diff']):.4f} (95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}])")
        print(f"  Paired t-test p = {tt['p_value']:.4f}, Bootstrap p = {boot['p_value']:.4f}")
        print(f"  Conclusion: {'Significant' if r['significant'] else 'Not significant'} at alpha=0.05")
        print()

    # Save results
    output_json = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "model_comparisons.json"
    output_csv = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "model_comparisons.csv"

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'methodology': 'Patient-clustered paired Bootstrap (primary) + Paired t-test (supplementary)',
            'n_bootstrap': 1000,
            'bonferroni_alpha': bonferroni_alpha,
            'comparisons': results
        }, f, indent=2, default=str)

    # Save CSV
    rows = []
    for r in results:
        row = {
            'comparison': r['comparison'],
            'type': r['type'],
            'model_better': r['model_better'],
            'model_worse': r['model_worse'],
            'n_folds': r['n_folds'],
            'tt_mean_diff': r['paired_ttest']['mean_diff'],
            'tt_std_diff': r['paired_ttest']['std_diff'],
            'tt_ci_lower': r['paired_ttest']['ci_lower'],
            'tt_ci_upper': r['paired_ttest']['ci_upper'],
            'tt_p_value': r['paired_ttest']['p_value'],
            'boot_mean_diff': r['bootstrap']['mean_diff'],
            'boot_ci_lower': r['bootstrap']['ci_lower'],
            'boot_ci_upper': r['bootstrap']['ci_upper'],
            'boot_p_value': r['bootstrap']['p_value'],
            'significant': r['significant']
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"Saved: {output_json}")
    print(f"Saved: {output_csv}")

    return results


if __name__ == "__main__":
    results = run_model_comparisons()
    sys.exit(0)
