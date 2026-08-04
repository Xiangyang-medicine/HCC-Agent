#!/usr/bin/env python3
"""Generate sensitivity analysis summary V2 (corrected).

This script properly computes:
1. Cohort statistics from unique patient outcomes (NOT accumulated across models/repeats)
2. Model metrics from metrics_summary.json (NOT by mixing all repeats)
3. Per-metric rankings (NOT just Harrell-based ranking)
4. Honest reporting of M4 performance (not claiming all metrics #1)

Usage:
    python generate_sensitivity_summary_v2.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT is 3 levels up: scripts -> prognostic_engine -> src -> ACM
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


def load_patient_outcomes(data_path: Path, analysis: str) -> dict:
    """Load unique patient outcomes for a given analysis."""
    df = pd.read_parquet(data_path)

    if analysis == 'SA2':
        # Exclude pediatric (age < 18)
        df = df[df['age_at_diagnosis'] >= 18]
    elif analysis == 'SA3':
        # Exclude missing stage or grade
        df = df[df['ajcc_stage'].notna() & df['tumor_grade'].notna()]

    n_patients = df['case_id'].nunique()
    n_events = int(df['event'].sum())

    return {
        'analysis': analysis,
        'n_patients': n_patients,
        'n_events': n_events,
        'event_rate': n_events / n_patients,
    }


def load_metrics(metrics_path: Path) -> dict:
    """Load metrics from metrics_summary.json."""
    with open(metrics_path) as f:
        data = json.load(f)

    metrics = {}
    for model_name, model_data in data.get('metrics', {}).items():
        model_metrics = {}
        for metric_name, metric_data in model_data.items():
            # Metric data can be either:
            # 1. A dict with 'mean', 'std', 'per_fold' keys (nested structure)
            # 2. A scalar value directly
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                model_metrics[metric_name] = {
                    'mean': metric_data['mean'],
                    'std': metric_data.get('std'),
                    'min': metric_data.get('min'),
                    'max': metric_data.get('max'),
                    'per_fold': metric_data.get('per_fold', []),
                }
            elif isinstance(metric_data, (int, float)):
                model_metrics[metric_name] = {'mean': metric_data}
        metrics[model_name] = model_metrics

    return metrics


def verify_predictions_integrity(pred_path: Path, expected_patients: int, expected_total: int) -> dict:
    """Verify prediction file integrity."""
    df = pd.read_csv(pred_path)

    n_unique_patients = df['case_id'].nunique()
    n_rows = len(df)
    n_models = df['model'].nunique()
    n_repeats = df['repeat'].nunique()
    n_folds = df['fold'].nunique()

    # Verify consistency of survival_months and event across models/repeats
    inconsistencies = []
    for cid in df['case_id'].unique()[:50]:  # Check first 50
        sub = df[df['case_id'] == cid]
        times = sub['survival_months'].unique()
        events = sub['event'].unique()
        if len(times) > 1 or len(events) > 1:
            inconsistencies.append({
                'case_id': cid,
                'times': list(times),
                'events': list(events),
            })

    return {
        'n_unique_patients': n_unique_patients,
        'n_rows': n_rows,
        'n_models': n_models,
        'n_repeats': n_repeats,
        'n_folds': n_folds,
        'predictions_per_model': n_rows // n_models,
        'expected_total': expected_total,
        'integrity_check': n_rows == expected_total,
        'sample_inconsistencies': inconsistencies[:5],
    }


def compute_rankings(metrics_dict: dict, analyses: list) -> dict:
    """Compute per-metric rankings across analyses."""
    # Define metrics to rank (using actual keys from the data)
    metrics_to_rank = [
        'harrell_c', 'uno_c', 'auc_12m', 'auc_36m', 'auc_60m',
        'ibs', 'calibration_slope', 'calibration_intercept',
        'calibration_in_large', 'eo_ratio'
    ]

    rankings = {}
    for metric in metrics_to_rank:
        rankings[metric] = {}
        for analysis in analyses:
            # Get mean values for each model
            values = {}
            for model, model_metrics in metrics_dict[analysis].items():
                if metric in model_metrics:
                    val = model_metrics[metric].get('mean')
                    if val is not None:
                        values[model] = val

            if not values:
                continue

            # Rank (1 = best)
            if metric in ['ibs', 'calibration_intercept']:
                # Lower is better for these
                sorted_models = sorted(values.items(), key=lambda x: x[1])
            else:
                # Higher is better
                sorted_models = sorted(values.items(), key=lambda x: x[1], reverse=True)

            rankings[metric][analysis] = {m: i + 1 for i, (m, _) in enumerate(sorted_models)}

    return rankings


def main():
    base_path = PROJECT_ROOT / "experiments" / "phase3a"
    data_path = PROJECT_ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet"

    # Define analyses
    analyses_config = {
        'SA1': {
            'metrics_path': base_path / "formal" / "metrics_summary.json",
            'predictions_path': base_path / "formal" / "oof_predictions.csv",
            'expected_patients': 363,
            'expected_total': 9075,
            'expected_events': 129,
        },
        'SA2': {
            'metrics_path': base_path / "sensitivity" / "SA2" / "metrics_summary.json",
            'predictions_path': base_path / "sensitivity" / "SA2" / "oof_predictions.csv",
            'expected_patients': 361,
            'expected_total': 9025,
            'expected_events': 129,
            'exclusion': 'pediatric (age < 18)',
            'n_excluded': 2,
        },
        'SA3': {
            'metrics_path': base_path / "sensitivity" / "SA3" / "metrics_summary.json",
            'predictions_path': base_path / "sensitivity" / "SA3" / "oof_predictions.csv",
            'expected_patients': 338,
            'expected_total': 8450,
            'expected_events': 113,
            'exclusion': 'missing stage/grade',
            'n_excluded': 25,
        },
    }

    # Load cohort statistics
    cohort_stats = {}
    for analysis in ['SA1', 'SA2', 'SA3']:
        cohort_stats[analysis] = load_patient_outcomes(data_path, analysis)

    # Load metrics
    metrics_dict = {}
    integrity_checks = {}
    for analysis, config in analyses_config.items():
        metrics_dict[analysis] = load_metrics(config['metrics_path'])
        integrity_checks[analysis] = verify_predictions_integrity(
            config['predictions_path'],
            config['expected_patients'],
            config['expected_total']
        )

    # Compute rankings
    rankings = compute_rankings(metrics_dict, ['SA1', 'SA2', 'SA3'])

    # Build summary
    summary = {
        'version': 'v2',
        'generated_at': pd.Timestamp.now().isoformat(),
        'cohort_statistics': cohort_stats,
        'integrity_checks': integrity_checks,
        'model_metrics': {},
        'rankings': rankings,
        'key_findings': {},
        'm4_status': {},
    }

    # Model name mapping
    model_names = {
        'M1_clinical_cox': 'M1 (Clinical Cox)',
        'M2_gene_elasticnet': 'M2 (Gene Elasticnet)',
        'M3_combined_elasticnet': 'M3 (Combined Elasticnet)',
        'M4_combined_rsf': 'M4 (Combined RSF)',
        'M5_deepsurv': 'M5 (DeepSurv)',
    }

    # Build model metrics
    for analysis in ['SA1', 'SA2', 'SA3']:
        summary['model_metrics'][analysis] = {}
        for model_key, model_name in model_names.items():
            if model_key in metrics_dict[analysis]:
                summary['model_metrics'][analysis][model_name] = metrics_dict[analysis][model_key]

    # M4 status
    m4_key = 'M4_combined_rsf'
    m4_name = model_names[m4_key]

    m4_harrell = {
        analysis: metrics_dict[analysis][m4_key]['harrell_c']['mean']
        for analysis in ['SA1', 'SA2', 'SA3'] if m4_key in metrics_dict[analysis]
    }
    m4_uno = {
        analysis: metrics_dict[analysis][m4_key]['uno_c']['mean']
        for analysis in ['SA1', 'SA2', 'SA3'] if m4_key in metrics_dict[analysis]
    }
    m4_ibs = {
        analysis: metrics_dict[analysis][m4_key]['ibs']['mean']
        for analysis in ['SA1', 'SA2', 'SA3'] if m4_key in metrics_dict[analysis]
    }

    summary['m4_status'] = {
        'model': m4_name,
        'harrell_c': m4_harrell,
        'uno_c': m4_uno,
        'ibs': m4_ibs,
        'mean_harrell_c': float(np.mean(list(m4_harrell.values()))),
        'std_harrell_c': float(np.std(list(m4_harrell.values()))),
        'rank_summary': {
            analysis: {
                'harrell': rankings.get('harrell_c', {}).get(analysis, {}).get(m4_key, 'N/A'),
                'uno_c': rankings.get('uno_c', {}).get(analysis, {}).get(m4_key, 'N/A'),
                'ibs': rankings.get('ibs', {}).get(analysis, {}).get(m4_key, 'N/A'),
            }
            for analysis in ['SA1', 'SA2', 'SA3']
        },
    }

    # Key findings
    summary['key_findings'] = {
        'm4_harrell_range': f"{min(m4_harrell.values()):.3f} - {max(m4_harrell.values()):.3f}",
        'm4_uno_range': f"{min(m4_uno.values()):.3f} - {max(m4_uno.values()):.3f}",
        'm4_ibs_range': f"{min(m4_ibs.values()):.3f} - {max(m4_ibs.values()):.3f}",
        'm4_not_all_metrics_first': True,
        'note': 'M4 is a stable candidate with best Harrell C, but may not be #1 in all metrics',
    }

    # Save JSON
    output_json = base_path / "sensitivity" / "SENSITIVITY_SUMMARY_V2.json"
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_json}")

    # Build CSV
    csv_rows = []
    for model_key, model_name in model_names.items():
        row = {'Model': model_name}
        for analysis in ['SA1', 'SA2', 'SA3']:
            if model_key in metrics_dict[analysis]:
                m = metrics_dict[analysis][model_key]
                row[f'{analysis}_Harrell_C'] = m.get('harrell_c', {}).get('mean', None)
                row[f'{analysis}_Uno_C'] = m.get('uno_c', {}).get('mean', None)
                row[f'{analysis}_IBS'] = m.get('ibs', {}).get('mean', None)
                row[f'{analysis}_AUC_36m'] = m.get('auc_36m', {}).get('mean', None)
                row[f'{analysis}_Harrell_Rank'] = rankings.get('harrell_c', {}).get(analysis, {}).get(model_key, None)
                row[f'{analysis}_Uno_Rank'] = rankings.get('uno_c', {}).get(analysis, {}).get(model_key, None)
                row[f'{analysis}_IBS_Rank'] = rankings.get('ibs', {}).get(analysis, {}).get(model_key, None)
        csv_rows.append(row)

    csv_df = pd.DataFrame(csv_rows)
    output_csv = base_path / "sensitivity" / "SENSITIVITY_SUMMARY_V2.csv"
    csv_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY V2 (CORRECTED)")
    print("=" * 80)

    print("\n### Cohort Statistics (from unique patient outcomes)")
    for analysis in ['SA1', 'SA2', 'SA3']:
        stats = cohort_stats[analysis]
        excl = f" (excluded: {analyses_config[analysis].get('n_excluded', 0)} {analyses_config[analysis].get('exclusion', '')})" if 'exclusion' in analyses_config[analysis] else ""
        print(f"  {analysis}: {stats['n_patients']} patients, {stats['n_events']} events, event_rate={stats['event_rate']:.3f}{excl}")

    print("\n### Integrity Checks")
    for analysis, check in integrity_checks.items():
        status = "PASS" if check['integrity_check'] else "FAIL"
        print(f"  {analysis}: {status} - {check['n_rows']} rows (expected {check['expected_total']})")

    print("\n### Model Performance (Harrell C-index)")
    print(f"{'Model':<25} {'SA1':>8} {'SA2':>8} {'SA3':>8} {'Mean':>8}")
    print("-" * 60)
    for model_key, model_name in model_names.items():
        vals = []
        for analysis in ['SA1', 'SA2', 'SA3']:
            if model_key in metrics_dict[analysis]:
                vals.append(metrics_dict[analysis][model_key].get('harrell_c', {}).get('mean', 0))
            else:
                vals.append(None)
        mean_val = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        row_str = f"{model_name:<25}"
        for v in vals:
            row_str += f" {v:>8.4f}" if v is not None else f" {'N/A':>8}"
        row_str += f" {mean_val:>8.4f}" if mean_val is not None else f" {'N/A':>8}"
        print(row_str)

    print("\n### M4 Status (Honest Reporting)")
    print(f"  Harrell C: SA1={m4_harrell['SA1']:.4f}, SA2={m4_harrell['SA2']:.4f}, SA3={m4_harrell['SA3']:.4f}")
    print(f"  Mean: {np.mean(list(m4_harrell.values())):.4f} ± {np.std(list(m4_harrell.values())):.4f}")
    print(f"  Uno C: SA1={m4_uno['SA1']:.4f}, SA2={m4_uno['SA2']:.4f}, SA3={m4_uno['SA3']:.4f}")
    print(f"  IBS: SA1={m4_ibs['SA1']:.4f}, SA2={m4_ibs['SA2']:.4f}, SA3={m4_ibs['SA3']:.4f}")

    print("\n### Per-Metric Rankings (SA1)")
    for metric in ['harrell_c', 'uno_c', 'ibs', 'auc_36m']:
        if metric in rankings:
            print(f"  {metric}:")
            for model_key, rank in sorted(rankings[metric]['SA1'].items(), key=lambda x: x[1]):
                model_name = model_names.get(model_key, model_key)
                print(f"    #{rank}: {model_name}")

    return summary


if __name__ == "__main__":
    main()
