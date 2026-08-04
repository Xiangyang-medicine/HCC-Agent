#!/usr/bin/env python3
"""Model comparisons V3 - Corrected implementation using existing bootstrap.py.

Key corrections vs v2:
1. Uses patient_level_paired_bootstrap from bootstrap.py (computes C-index properly)
2. Supplementary t-test uses per-fold metrics from metrics_summary.json (not OOF mixing)
3. Adds both Harrell C and Uno C comparisons

Usage:
    python run_model_comparisons_v3.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Import from existing bootstrap module
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROGNOSTIC_SRC = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(PROGNOSTIC_SRC))
from prognostic_engine.bootstrap import patient_level_paired_bootstrap
from prognostic_engine.metrics import harrell_c_index

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase3a" / "formal"


def load_metrics_summary(metrics_path: Path) -> dict:
    """Load metrics from metrics_summary.json."""
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get('metrics', {})


def paired_ttest_per_fold(
    metrics_a: list,
    metrics_b: list
) -> dict:
    """Paired t-test on per-fold metrics from metrics_summary.json.

    This is the CORRECT supplementary analysis - uses the 25 per-fold
    metric values already computed during cross-validation.
    """
    if len(metrics_a) != len(metrics_b):
        raise ValueError(f"Metric lists must have same length: {len(metrics_a)} vs {len(metrics_b)}")

    metrics_a = np.array(metrics_a)
    metrics_b = np.array(metrics_b)

    diffs = metrics_a - metrics_b
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se = std_diff / np.sqrt(len(diffs))

    # Two-sided paired t-test
    t_stat, p_value = stats.ttest_rel(metrics_a, metrics_b)

    # 95% CI
    ci_lower = mean_diff - stats.t.ppf(0.975, len(diffs) - 1) * se
    ci_upper = mean_diff + stats.t.ppf(0.975, len(diffs) - 1) * se

    return {
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'se': float(se),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_pairs': len(diffs),
    }


def main():
    print("=" * 80)
    print("MODEL COMPARISONS V3 - CORRECTED IMPLEMENTATION")
    print("=" * 80)
    print()

    # Load data
    oof_path = OUTPUT_DIR / "oof_predictions.csv"
    metrics_path = OUTPUT_DIR / "metrics_summary.json"

    print(f"Loading OOF predictions from {oof_path}...")
    oof_df = pd.read_csv(oof_path)
    print(f"  Loaded {len(oof_df)} rows, {oof_df['case_id'].nunique()} patients")

    print(f"Loading metrics from {metrics_path}...")
    metrics_summary = load_metrics_summary(metrics_path)
    print(f"  Loaded metrics for {len(metrics_summary)} models")

    # Define model mappings
    model_keys = {
        'M1_clinical_cox': 'M1 (Clinical Cox)',
        'M2_gene_elasticnet': 'M2 (Gene Elasticnet)',
        'M3_combined_elasticnet': 'M3 (Combined Elasticnet)',
        'M4_combined_rsf': 'M4 (Combined RSF)',
        'M5_deepsurv': 'M5 (DeepSurv)',
    }

    # Define comparisons
    comparisons = [
        # (model_a_key, model_b_key, comparison_type)
        ('M3_combined_elasticnet', 'M1_clinical_cox', 'Formal'),
        ('M4_combined_rsf', 'M1_clinical_cox', 'Formal'),
        ('M5_deepsurv', 'M1_clinical_cox', 'Formal'),
        ('M3_combined_elasticnet', 'M2_gene_elasticnet', 'Formal'),
        ('M4_combined_rsf', 'M2_gene_elasticnet', 'Exploratory'),
    ]

    # Run comparisons for both metrics
    all_results = {
        'timestamp_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'methodology': 'Patient-level paired bootstrap (primary) + Per-fold paired t-test (supplementary)',
        'n_bootstrap_iterations': 1000,
        'bonferroni_alpha': 0.05,
        'n_formal_comparisons': 4,
        'harrell_c_comparisons': [],
        'uno_c_comparisons': [],
    }

    for metric_name in ['harrell_c', 'uno_c']:
        metric_display = 'Harrell C' if metric_name == 'harrell_c' else 'Uno C'
        results_key = f'{metric_name}_comparisons'
        metric_func = harrell_c_index if metric_name == 'harrell_c' else None  # Uno C requires IPCW training data

        print()
        print(f"### {metric_display}-index Comparisons ###")
        print()

        for model_a_key, model_b_key, comp_type in comparisons:
            model_a_name = model_keys[model_a_key]
            model_b_name = model_keys[model_b_key]

            print(f"  Comparing {model_a_name} vs {model_b_name} ({metric_name})...")

            # Primary: Patient-level bootstrap (Harrell C only - Uno C requires IPCW training data)
            if metric_name == 'harrell_c':
                try:
                    boot_result = patient_level_paired_bootstrap(
                        oof_df,
                        n_iterations=1000,
                        seed=456,
                        metric_func=metric_func,
                        comparison_pair=(model_a_key, model_b_key)
                    )
                except Exception as e:
                    print(f"    ERROR in bootstrap: {e}")
                    boot_result = {'status': 'ERROR', 'error': str(e)}
            else:
                # Uno C bootstrap not possible without training IPCW data
                # Use observed mean difference from metrics_summary.json
                mean_a = metrics_summary.get(model_a_key, {}).get(metric_name, {}).get('mean', 0)
                mean_b = metrics_summary.get(model_b_key, {}).get(metric_name, {}).get('mean', 0)
                boot_result = {
                    'status': 'SUPPLEMENTARY_ONLY',
                    'note': 'Uno C requires IPCW training data not available in OOF predictions',
                    'observed_mean_difference': mean_a - mean_b,
                    'p_value': None,
                    'ci_lower': None,
                    'ci_upper': None,
                }

            # Supplementary: Per-fold paired t-test using metrics_summary
            metrics_a = metrics_summary.get(model_a_key, {}).get(metric_name, {}).get('per_fold', [])
            metrics_b = metrics_summary.get(model_b_key, {}).get(metric_name, {}).get('per_fold', [])

            if len(metrics_a) > 0 and len(metrics_b) > 0:
                tt_result = paired_ttest_per_fold(metrics_a, metrics_b)
            else:
                print(f"    WARNING: Missing {metric_name} per_fold data")
                tt_result = None

            # Determine which model is better (based on observed mean)
            mean_a = metrics_summary.get(model_a_key, {}).get(metric_name, {}).get('mean', 0)
            mean_b = metrics_summary.get(model_b_key, {}).get(metric_name, {}).get('mean', 0)

            if mean_a > mean_b:
                model_better, model_worse = model_a_name, model_b_name
            else:
                model_better, model_worse = model_b_name, model_a_name

            # Adjusted p-value (Bonferroni for formal comparisons)
            if boot_result.get('status') == 'SUCCESS':
                raw_p = boot_result.get('p_value', 1.0)
                if comp_type == 'Formal':
                    p_adj = min(raw_p * 4, 1.0)  # 4 formal comparisons
                else:
                    p_adj = raw_p  # Exploratory
                significant = p_adj < 0.05
            elif boot_result.get('status') == 'SUPPLEMENTARY_ONLY' and tt_result:
                # Use t-test p-value for Uno C
                raw_p = tt_result['p_value']
                if comp_type == 'Formal':
                    p_adj = min(raw_p * 4, 1.0)
                else:
                    p_adj = raw_p
                significant = p_adj < 0.05
            else:
                raw_p = 1.0
                p_adj = 1.0
                significant = False

            result = {
                'comparison': f"{model_a_name} vs {model_b_name}",
                'type': comp_type,
                'metric': metric_name,
                'model_better': model_better,
                'model_worse': model_worse,
                'patient_bootstrap': {
                    'status': boot_result.get('status', 'UNKNOWN'),
                    'mean_diff': boot_result.get('observed_mean_difference', boot_result.get('mean_diff', 0)),
                    'std_diff': None,
                    'ci_lower': boot_result.get('ci_lower'),
                    'ci_upper': boot_result.get('ci_upper'),
                    'p_value_raw': boot_result.get('p_value'),
                    'p_value_adjusted': float(p_adj),
                    'iterations_valid': boot_result.get('iterations_valid') if boot_result.get('status') == 'SUCCESS' else None,
                    'methodology': 'patient_level_paired_bootstrap' if metric_name == 'harrell_c' else 'supplementary_ttest_only',
                },
                'paired_ttest_supplementary': tt_result,
                'significant': significant,
                'n_patients': oof_df['case_id'].nunique(),
                'n_folds': 25,
                'observed_means': {
                    model_a_name: float(mean_a),
                    model_b_name: float(mean_b),
                }
            }

            all_results[results_key].append(result)

    # Save JSON
    json_path = OUTPUT_DIR / "model_comparisons_v3.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Create CSV
    csv_rows = []
    for results_key in ['harrell_c_comparisons', 'uno_c_comparisons']:
        metric = 'harrell_c' if results_key == 'harrell_c_comparisons' else 'uno_c'
        for r in all_results[results_key]:
            boot = r['patient_bootstrap']
            tt = r['paired_ttest_supplementary']
            csv_rows.append({
                'comparison': r['comparison'],
                'type': r['type'],
                'metric': metric,
                'model_better': r['model_better'],
                'model_worse': r['model_worse'],
                'n_patients': r['n_patients'],
                'model_a_mean': r['observed_means'].get(r['comparison'].split(' vs ')[0], None),
                'model_b_mean': r['observed_means'].get(r['comparison'].split(' vs ')[1], None),
                'boot_mean_diff': boot.get('mean_diff'),
                'boot_ci_lower': boot.get('ci_lower'),
                'boot_ci_upper': boot.get('ci_upper'),
                'boot_p_value_raw': boot.get('p_value_raw'),
                'boot_p_value_adj': boot.get('p_value_adjusted'),
                'tt_mean_diff': tt['mean_diff'] if tt else None,
                'tt_t_stat': tt['t_statistic'] if tt else None,
                'tt_p_value': tt['p_value'] if tt else None,
                'significant': r['significant'],
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUTPUT_DIR / "model_comparisons_v3.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for metric, results_key in [('Harrell C', 'harrell_c_comparisons'), ('Uno C', 'uno_c_comparisons')]:
        print()
        print(f"### {metric}-index Results ###")
        print(f"{'Comparison':<35} {'Type':<12} {'Mean A':>8} {'Mean B':>8} {'Diff':>8} {'p_adj':>8} {'Sig':>5}")
        print("-" * 85)
        for r in all_results[results_key]:
            boot = r['patient_bootstrap']
            means = r['observed_means']
            model_a_mean = means.get(r['comparison'].split(' vs ')[0], 0)
            model_b_mean = means.get(r['comparison'].split(' vs ')[1], 0)
            diff = model_a_mean - model_b_mean
            sig = "***" if r['significant'] else ""
            print(f"{r['comparison']:<35} {r['type']:<12} {model_a_mean:>8.4f} {model_b_mean:>8.4f} {diff:>8.4f} {boot['p_value_adjusted']:>8.4f} {sig:>5}")

    print()
    print("### Supplementary T-Test (Per-Fold, n=25) ###")
    for metric, results_key in [('Harrell C', 'harrell_c_comparisons'), ('Uno C', 'uno_c_comparisons')]:
        print(f"\n{metric}:")
        for r in all_results[results_key]:
            tt = r['paired_ttest_supplementary']
            if tt:
                print(f"  {r['comparison']}: mean_diff={tt['mean_diff']:+.4f}, t={tt['t_statistic']:.3f}, p={tt['p_value']:.4e}")

    # Note about v2 supersession
    print()
    print("=" * 80)
    print("NOTE: model_comparisons_v2.json/csv are SUPERSEDED (contained errors)")
    print("      - tt_mean_diff=31.78 was impossible for C-index (0-1 range)")
    print("      - Used wrong methodology for supplementary analysis")
    print("      - Please use model_comparisons_v3.json/csv")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
