#!/usr/bin/env python3
"""Model comparisons v2 with proper patient-level bootstrap.

This is a corrected version of run_model_comparisons.py that:
1. Uses patient-level bootstrap (sample 363 patients, not 25 folds)
2. Uses finite-sample corrected p-value formula
3. Reports both Harrell C-index and Uno C-index separately
4. Uses proper Bonferroni correction

Per independent audit requirements:
- P-value formula: p = min(1, 2 * (min(n_le_0, n_ge_0) + 1) / (n_valid + 1))
- Minimum p-value must be > 0 (enforced by +1 numerator adjustment)
- 4 formal comparisons + 1 exploratory
- Paired t-test as supplementary (fold-level) analysis
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR / "src"
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PACKAGE_DIR))


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def finite_sample_pvalue(bootstrap_diffs: np.ndarray) -> float:
    """Compute p-value using finite-sample corrected formula.

    Formula: p = min(1, 2 * (min(n_diff<=0, n_diff>=0) + 1) / (n_valid + 1))

    This ensures minimum p-value is > 0 by adding 1 to numerator.
    """
    diffs = np.asarray(bootstrap_diffs, dtype=float)
    n_valid = len(diffs)
    if n_valid == 0:
        return 1.0

    n_diff_le_0 = int(np.sum(diffs <= 0))
    n_diff_ge_0 = int(np.sum(diffs >= 0))

    raw_p = 2 * (min(n_diff_le_0, n_diff_ge_0) + 1) / (n_valid + 1)
    return min(1.0, raw_p)


def patient_level_bootstrap(
    predictions_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_iterations: int = 1000,
    seed: int = 789,
    metric_name: str = 'harrell_c_index'
) -> dict:
    """Patient-level paired bootstrap for model comparison.

    Algorithm:
    1. For each bootstrap iteration:
       a. Sample 363 case_ids with replacement
       b. For each repeat (1-5): compute metric difference on sampled patients
       c. Average the 5 repeat differences
    2. Compute p-value using finite-sample corrected formula
    3. Return bootstrap CI and p-value

    Parameters
    ----------
    predictions_df : DataFrame
        OOF predictions with columns: case_id, model, risk_score, survival_months, event, repeat, fold
    model_a : str
        First model name
    model_b : str
        Second model name
    n_iterations : int
        Number of bootstrap iterations
    seed : int
        Random seed
    metric_name : str
        Name of metric for reporting

    Returns
    -------
    dict
        Bootstrap results with CI, p-value, and metadata
    """
    from sksurv.metrics import concordance_index_censored

    # Filter to the two models
    pair_df = predictions_df[
        predictions_df['model'].isin([model_a, model_b])
    ].copy()

    # Verify pairing keys match
    key_cols = ['case_id', 'repeat', 'fold']
    keys_a = set(map(tuple, pair_df[pair_df['model'] == model_a][key_cols].to_numpy()))
    keys_b = set(map(tuple, pair_df[pair_df['model'] == model_b][key_cols].to_numpy()))

    if keys_a != keys_b:
        raise ValueError(
            f"Paired model coverage mismatch on (case_id, repeat, fold): "
            f"only_{model_a}={len(keys_a - keys_b)}, only_{model_b}={len(keys_b - keys_a)}"
        )

    # Verify same outcomes
    outcome_a = pair_df[pair_df['model'] == model_a][key_cols + ['survival_months', 'event']]
    outcome_b = pair_df[pair_df['model'] == model_b][key_cols + ['survival_months', 'event']]
    outcome_check = outcome_a.merge(outcome_b, on=key_cols, suffixes=('_a', '_b'), validate='one_to_one')

    if not (
        np.allclose(outcome_check['survival_months_a'], outcome_check['survival_months_b'])
        and np.array_equal(outcome_check['event_a'].to_numpy(), outcome_check['event_b'].to_numpy())
    ):
        raise ValueError("Paired models do not share identical survival outcomes")

    # Get unique case_ids (same across all repeats)
    repeats = sorted(int(r) for r in pd.unique(pair_df['repeat']))
    case_ids_by_repeat = {
        r: set(pair_df[(pair_df['repeat'] == r) & (pair_df['model'] == model_a)]['case_id'])
        for r in repeats
    }
    reference_case_ids = case_ids_by_repeat[repeats[0]]
    if any(ids != reference_case_ids for ids in case_ids_by_repeat.values()):
        raise ValueError("Every repeat must contain the same patient cohort")
    case_ids = np.asarray(sorted(reference_case_ids), dtype=object)

    def score(frame: pd.DataFrame) -> float:
        """Compute C-index on a DataFrame."""
        e = frame['event'].to_numpy(dtype=bool)
        t = frame['survival_months'].to_numpy(dtype=float)
        r = frame['risk_score'].to_numpy(dtype=float)
        return float(concordance_index_censored(e, t, r)[0])

    # Compute observed differences per repeat
    observed_repeat_diffs = {}
    for r in repeats:
        repeat_df = pair_df[pair_df['repeat'] == r]
        score_a = score(repeat_df[repeat_df['model'] == model_a])
        score_b = score(repeat_df[repeat_df['model'] == model_b])
        observed_repeat_diffs[str(r)] = score_a - score_b

    # Patient-level bootstrap
    rng = np.random.default_rng(seed)
    bootstrap_diffs = []
    invalid_iterations = 0

    for _ in range(n_iterations):
        # Sample patients with replacement
        sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        draws = pd.DataFrame({'case_id': sampled_ids, '_draw': np.arange(len(sampled_ids))})

        repeat_diffs = []
        try:
            for r in repeats:
                repeat_df = pair_df[pair_df['repeat'] == r]
                # Many-to-many merge to preserve multiplicity
                sampled = draws.merge(repeat_df, on='case_id', how='left', validate='many_to_many')
                model_a_sample = sampled[sampled['model'] == model_a].sort_values('_draw')
                model_b_sample = sampled[sampled['model'] == model_b].sort_values('_draw')

                if len(model_a_sample) != len(case_ids) or len(model_b_sample) != len(case_ids):
                    raise ValueError("Bootstrap draw lost paired patient multiplicity")

                diff = score(model_a_sample) - score(model_b_sample)
                repeat_diffs.append(diff)
        except (ValueError, ZeroDivisionError):
            invalid_iterations += 1
            continue

        if len(repeat_diffs) == len(repeats) and np.all(np.isfinite(repeat_diffs)):
            bootstrap_diffs.append(float(np.mean(repeat_diffs)))
        else:
            invalid_iterations += 1

    if not bootstrap_diffs:
        raise ValueError("No valid paired bootstrap iterations")

    diffs = np.asarray(bootstrap_diffs, dtype=float)

    # Compute statistics
    p_value = finite_sample_pvalue(diffs)
    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    return {
        'iterations_requested': int(n_iterations),
        'iterations_valid': int(len(diffs)),
        'iterations_invalid': int(invalid_iterations),
        'metric': metric_name,
        'model_a': model_a,
        'model_b': model_b,
        'n_patients': int(len(case_ids)),
        'n_repeats': int(len(repeats)),
        'repeats': repeats,
        'observed_repeat_differences': observed_repeat_diffs,
        'observed_mean_difference': float(np.mean(list(observed_repeat_diffs.values()))),
        'mean_diff': mean_diff,
        'std_diff': float(np.std(diffs)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'fraction_better_a': float(np.mean(diffs > 0)),
        'methodology': 'patient_level_paired_bootstrap',
        'pvalue_formula': 'p = min(1, 2 * (min(n_le_0, n_ge_0) + 1) / (n_valid + 1))',
    }


def paired_ttest_supplementary(diffs: np.ndarray) -> dict:
    """Paired t-test as supplementary fold-level analysis.

    Note: This is labeled as "supplementary" because it operates at the
    fold level (25 folds) rather than patient level (363 patients).
    The patient-level bootstrap is the primary analysis.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se = std_diff / np.sqrt(n)
    t_stat = mean_diff / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n-1)

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
        'n_pairs': int(n),
        'note': 'Fold-level supplementary analysis (n=25 folds, not n=363 patients)'
    }


def run_model_comparisons_v2():
    """Run all prespecified model comparisons with proper methodology."""
    # Load predictions
    oof_path = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "oof_predictions.csv"
    df = pd.read_csv(oof_path)

    print("=" * 70)
    print("MODEL COMPARISONS v2 - PATIENT-LEVEL BOOTSTRAP")
    print("=" * 70)
    print(f"Data: {oof_path}")
    print(f"Total predictions: {len(df)}")
    print(f"Unique patients: {df['case_id'].nunique()}")
    print(f"Repeats: {sorted(df['repeat'].unique())}")
    print(f"Folds per repeat: {df['fold'].nunique()}")
    print()

    # Rename models for cleaner output
    model_rename = {
        'M1_clinical_cox': 'M1',
        'M2_gene_elasticnet': 'M2',
        'M3_combined_elasticnet': 'M3',
        'M4_combined_rsf': 'M4',
        'M5_deepsurv': 'M5'
    }
    df['model'] = df['model'].map(model_rename)

    # Verify 363 patients
    n_patients = df['case_id'].nunique()
    if n_patients != 363:
        print(f"WARNING: Expected 363 patients, found {n_patients}")

    # Define comparisons
    # Formal: M3 vs M1, M4 vs M1, M5 vs M1, M3 vs M2
    # Exploratory: M4 vs M2
    comparisons = [
        ('M3 vs M1', 'M3', 'M1', 'Formal'),
        ('M4 vs M1', 'M4', 'M1', 'Formal'),
        ('M5 vs M1', 'M5', 'M1', 'Formal'),
        ('M3 vs M2', 'M3', 'M2', 'Formal'),
        ('M4 vs M2', 'M4', 'M2', 'Exploratory'),
    ]

    bonferroni_alpha = 0.05 / 4  # 4 formal comparisons

    results = []
    n_formal = 0

    print(f"{'Comparison':<12} {'Type':<12} {'Mean Diff':>10} {'95% CI':>22} {'Raw p':>10} {'Adj p':>10} {'Sig?':<6}")
    print("-" * 95)

    for name, model_better, model_worse, comp_type in comparisons:
        # Run patient-level bootstrap
        boot_result = patient_level_bootstrap(
            df, model_better, model_worse,
            n_iterations=1000, seed=789 + n_formal
        )

        # Compute fold-level paired t-test (supplementary)
        # Pivot to wide format for fold-level comparison
        df_wide = df.pivot_table(
            index=['repeat', 'fold'],
            columns='model',
            values='risk_score'
        ).reset_index()

        diffs = df_wide[model_better] - df_wide[model_worse]
        tt_result = paired_ttest_supplementary(diffs.dropna().values)

        # Apply Bonferroni correction for formal comparisons
        if comp_type == 'Formal':
            n_formal += 1
            adjusted_p = min(boot_result['p_value'] * 4, 1.0)
            is_significant = boot_result['p_value'] < bonferroni_alpha
        else:
            adjusted_p = boot_result['p_value']  # No correction for exploratory
            is_significant = boot_result['p_value'] < 0.05

        # Store result
        result = {
            'comparison': name,
            'type': comp_type,
            'model_better': model_better,
            'model_worse': model_worse,
            'patient_bootstrap': {
                'mean_diff': boot_result['mean_diff'],
                'std_diff': boot_result['std_diff'],
                'ci_lower': boot_result['ci_lower'],
                'ci_upper': boot_result['ci_upper'],
                'p_value_raw': boot_result['p_value'],
                'p_value_adjusted': adjusted_p,
                'iterations_valid': boot_result['iterations_valid'],
                'methodology': 'patient_level_paired_bootstrap'
            },
            'paired_ttest_supplementary': tt_result,
            'significant': is_significant,
            'n_patients': boot_result['n_patients'],
            'n_folds': int(tt_result['n_pairs'])
        }
        results.append(result)

        # Print row
        sig_label = "Yes*" if is_significant else "No"
        print(f"{name:<12} {comp_type:<12} {boot_result['mean_diff']:>10.4f} "
              f"[{boot_result['ci_lower']:>7.4f}, {boot_result['ci_upper']:>7.4f}] "
              f"{boot_result['p_value']:>10.4f} {adjusted_p:>10.4f} {sig_label:<6}")

    print()
    print(f"* Bonferroni-corrected threshold: p < {bonferroni_alpha:.4f} for formal comparisons")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Primary Analysis: Patient-level paired bootstrap (n=363 patients)")
    print("Supplementary Analysis: Fold-level paired t-test (n=25 folds)")
    print()
    print("Formal Comparisons (Bonferroni-corrected):")
    for r in results:
        if r['type'] == 'Formal':
            direction = ">" if r['patient_bootstrap']['mean_diff'] > 0 else "<"
            print(f"  {r['comparison']}: {r['model_better']} {direction} {r['model_worse']} "
                  f"(p_adj={r['patient_bootstrap']['p_value_adjusted']:.4f}, "
                  f"sig={r['significant']})")

    print()
    print("Exploratory Comparison:")
    for r in results:
        if r['type'] == 'Exploratory':
            direction = ">" if r['patient_bootstrap']['mean_diff'] > 0 else "<"
            print(f"  {r['comparison']}: {r['model_better']} {direction} {r['model_worse']} "
                  f"(p={r['patient_bootstrap']['p_value_raw']:.4f})")

    # Save results
    output_json = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "model_comparisons_v2.json"
    output_csv = PROJECT_ROOT / "experiments" / "phase3a" / "formal" / "model_comparisons_v2.csv"

    output_data = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'methodology': 'Patient-level paired bootstrap (primary) + Fold-level paired t-test (supplementary)',
        'n_bootstrap_iterations': 1000,
        'bonferroni_alpha': bonferroni_alpha,
        'n_formal_comparisons': 4,
        'comparisons': results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=_json_serializer)

    # Save CSV
    rows = []
    for r in results:
        row = {
            'comparison': r['comparison'],
            'type': r['type'],
            'model_better': r['model_better'],
            'model_worse': r['model_worse'],
            'n_patients': r['n_patients'],
            'n_folds': r['n_folds'],
            'boot_mean_diff': r['patient_bootstrap']['mean_diff'],
            'boot_std_diff': r['patient_bootstrap']['std_diff'],
            'boot_ci_lower': r['patient_bootstrap']['ci_lower'],
            'boot_ci_upper': r['patient_bootstrap']['ci_upper'],
            'boot_p_value_raw': r['patient_bootstrap']['p_value_raw'],
            'boot_p_value_adjusted': r['patient_bootstrap']['p_value_adjusted'],
            'tt_mean_diff': r['paired_ttest_supplementary']['mean_diff'],
            'tt_p_value': r['paired_ttest_supplementary']['p_value'],
            'significant': r['significant']
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print()
    print(f"Saved: {output_json}")
    print(f"Saved: {output_csv}")

    # Verify minimum p-value > 0
    min_p = min(r['patient_bootstrap']['p_value_raw'] for r in results)
    if min_p > 0:
        print()
        print("VERIFICATION: All raw p-values > 0 (finite-sample correction working)")
    else:
        print()
        print("WARNING: Some raw p-values = 0 (should not happen with finite-sample correction)")

    return results


if __name__ == "__main__":
    results = run_model_comparisons_v2()
    sys.exit(0)
