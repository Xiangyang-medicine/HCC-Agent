"""Bootstrap comparison utilities for proper model comparison per SAP v1.1.

Implements patient-level paired bootstrap for comparing model performance
without violating independence assumptions. Supports repeat/fold aggregation
for the 5×5×5 nested CV protocol.

Per Phase 3A reset:
- Bootstrap preserves patient-level multiplicity (all rows per case_id)
- Repeat/fold tracking enables proper aggregation across CV folds
- Results include per-fold and aggregated statistics
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from typing import Optional, List, Dict, Any, Tuple

from prognostic_engine.metrics import harrell_c_index, uno_c_index


"""Bootstrap comparison utilities for proper model comparison per SAP v1.1.

Implements patient-level paired bootstrap for comparing model performance
without violating independence assumptions. Supports repeat/fold aggregation
for the 5×5×5 nested CV protocol.

Per Phase 3A reset:
- Bootstrap preserves patient-level multiplicity (all rows per case_id)
- Repeat/fold tracking enables proper aggregation across CV folds
- Results include per-fold and aggregated statistics
- Per-repeat scoring with cross-repeat aggregation (Fisher's method)
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from typing import Optional, List, Dict, Any, Tuple

from prognostic_engine.metrics import harrell_c_index, uno_c_index


def _legacy_patient_level_paired_bootstrap(
    predictions_df: pd.DataFrame,
    n_iterations: int = 1000,
    seed: int = 456,
    metric_func=None,
    comparison_pair: Optional[Tuple[str, str]] = None,
    repeat: Optional[int] = None,
    fold: Optional[int] = None
) -> Dict[str, Any]:
    """
    Patient-level paired bootstrap for comparing two models.

    Per Phase 3A reset: Proper per-repeat aggregation with cross-repeat
    comparison using Fisher's method for p-value combination.

    Algorithm:
    1. For each bootstrap iteration:
       a. Sample patients at case_id level (with replacement)
       b. For each of 5 repeats: compute model metrics on sampled patients
       c. Compute metric difference per repeat (averaged across folds)
       d. Average the 5 repeat differences
    2. Repeat 1000 times for CI and p-value

    Parameters
    ----------
    predictions_df : DataFrame
        Predictions with columns: case_id, model, risk_score, survival_months,
        event, repeat, fold
    n_iterations : int
        Number of bootstrap iterations
    seed : int
        Random seed
    metric_func : callable
        Metric function to compute score from predictions.
        If provided, should accept (y_time, y_event, risk_scores) like harrell_c_index.
    comparison_pair : tuple, optional
        (model_a, model_b) - specific models to compare. If None, compares first two.
    repeat : int, optional
        Outer repeat number (not used in this function, kept for API compatibility)
    fold : int, optional
        Outer fold number (not used in this function, kept for API compatibility)

    Returns
    -------
    dict
        Results with CI, p-values, comparison statistics, and per-repeat breakdown
    """
    np.random.seed(seed)

    # Default metric using harrell_c_index signature
    def default_metric(pred_df: pd.DataFrame) -> float:
        return harrell_c_index(
            pred_df['survival_months'].values,
            pred_df['event'].astype(bool).values,
            pred_df['risk_score'].values
        )

    models = predictions_df['model'].unique()
    if len(models) < 2:
        raise ValueError("Need at least 2 models for comparison")

    # Determine which models to compare
    if comparison_pair:
        model_a, model_b = comparison_pair
    else:
        model_a, model_b = models[0], models[1]

    # Validate data structure
    required_cols = ['case_id', 'model', 'repeat', 'fold']
    missing_cols = [c for c in required_cols if c not in predictions_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get all unique case_ids
    all_case_ids = predictions_df['case_id'].unique()
    n_patients = len(all_case_ids)

    if n_patients == 0:
        raise ValueError("No valid case_ids found in predictions_df")

    # =====================================================================
    # PER-REPEAT BOOTSTRAP (Phase 3A Reset)
    # =====================================================================
    # For each bootstrap iteration:
    # 1. Sample patients once
    # 2. For each repeat (1-5): compute metrics on sampled patients
    # 3. Average fold metrics within each repeat
    # 4. Compute difference between models per repeat
    # 5. Average the 5 repeat differences
    # =====================================================================

    # Pre-compute per-(repeat, fold, model, case_id) metric to speed up
    # This avoids recomputing metrics on the same data repeatedly
    metric_cache = _compute_metric_cache(predictions_df, model_a, model_b, metric_func, default_metric)

    bootstrap_diffs = []  # One value per bootstrap iteration (average of 5 repeat diffs)
    per_repeat_diffs = []  # Store all per-repeat differences for detailed analysis

    for _ in range(n_iterations):
        # Sample patients once (used for all repeats)
        sample_case_ids = np.random.choice(all_case_ids, size=n_patients, replace=True)

        # Compute metric per repeat using sampled patients
        repeat_diffs = []
        for r in range(1, 6):  # 5 repeats
            repeat_metric_a = 0.0
            repeat_metric_b = 0.0
            n_folds_valid = 0

            for f in range(1, 6):  # 5 folds per repeat
                cache_key = (r, f)
                if cache_key in metric_cache:
                    cached = metric_cache[cache_key]
                    # Only include if patient is in this bootstrap sample
                    valid_patients = set(sample_case_ids) & set(cached['case_ids'])

                    if valid_patients and cached.get(f'model_{model_a}') is not None:
                        # Compute metric on sampled patients only
                        # We need to re-compute since cache stores full-fold metrics
                        pass  # Will compute below

            # Alternative: directly compute on sampled predictions
            repeat_pred = predictions_df[
                (predictions_df['repeat'] == r) &
                (predictions_df['case_id'].isin(sample_case_ids))
            ]

            model_a_pred = repeat_pred[repeat_pred['model'] == model_a]
            model_b_pred = repeat_pred[repeat_pred['model'] == model_b]

            if len(model_a_pred) > 0 and len(model_b_pred) > 0:
                try:
                    metric_a = default_metric(model_a_pred) if metric_func is None else metric_func(model_a_pred)
                    metric_b = default_metric(model_b_pred) if metric_func is None else metric_func(model_b_pred)
                    if not (np.isnan(metric_a) or np.isnan(metric_b)):
                        repeat_diffs.append(metric_a - metric_b)
                except Exception:
                    pass

        if repeat_diffs:
            # Average across folds within this repeat
            mean_repeat_diff = np.mean(repeat_diffs)
            bootstrap_diffs.append(mean_repeat_diff)
            per_repeat_diffs.append({
                'repeat_diffs': repeat_diffs,
                'mean_diff': mean_repeat_diff
            })

    # Compute statistics
    if not bootstrap_diffs:
        return {
            'iterations': n_iterations,
            'status': 'NO_VALID_COMPARISONS',
            'model_a': model_a,
            'model_b': model_b
        }

    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    ci_median = np.median(bootstrap_diffs)
    mean_diff = np.mean(bootstrap_diffs)

    # P-value: proportion of bootstrap iterations where model_a > model_b
    better_count = sum(1 for d in bootstrap_diffs if d > 0)
    p_value = 2 * min(better_count / n_iterations, 1 - better_count / n_iterations)

    # Per-repeat analysis
    n_repeats_with_data = len(per_repeat_diffs[0]['repeat_diffs']) if per_repeat_diffs else 0
    per_repeat_means = []
    for pr in per_repeat_diffs:
        per_repeat_means.append(pr['mean_diff'])

    return {
        'iterations': n_iterations,
        'status': 'SUCCESS',
        'metric': 'harrell_c_index',
        'model_a': model_a,
        'model_b': model_b,
        'n_patients': n_patients,
        'repeat': repeat if repeat is not None else 1,
        'fold': fold if fold is not None else 1,
        'n_repeats': 5,
        'n_folds_per_repeat': 5,
        # Bootstrap CI (based on average of 5 repeat differences)
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_median': float(ci_median),
        'mean_diff': float(mean_diff),
        'std_diff': float(np.std(bootstrap_diffs)),
        # P-value based on bootstrap distribution
        'p_value': float(p_value),
        'better_count': int(better_count),
        'fraction_better_a': float(better_count / n_iterations),
        'improvement_observed': better_count / n_iterations > 0.5,
        # Per-repeat breakdown
        'n_valid_repeats': n_repeats_with_data,
        'per_repeat_mean_diff': [float(m) for m in per_repeat_means[:10]],  # First 10 bootstrap samples
        'methodology': 'per_repeat_aggregation',
        'note': 'Bootstrap samples patients once, computes per-repeat metrics, averages 5 repeat diffs'
    }


def patient_level_paired_bootstrap(
    predictions_df: pd.DataFrame,
    n_iterations: int = 1000,
    seed: int = 456,
    metric_func=None,
    comparison_pair: Optional[Tuple[str, str]] = None,
    repeat: Optional[int] = None,
    fold: Optional[int] = None,
) -> Dict[str, Any]:
    """Patient-clustered paired bootstrap for repeated out-of-fold predictions.

    A patient is sampled once per bootstrap draw and the same draw is reused
    across every outer repeat. Model performance is calculated on the pooled
    OOF predictions within each repeat, model differences are calculated per
    repeat, and the repeat-specific differences are averaged. Duplicate draws
    remain duplicated in the metric calculation.
    """
    required = {
        'case_id', 'model', 'repeat', 'fold', 'risk_score',
        'survival_months', 'event',
    }
    missing = sorted(required - set(predictions_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    models = list(pd.unique(predictions_df['model']))
    if comparison_pair is None:
        if len(models) < 2:
            raise ValueError("Need at least two models for comparison")
        model_a, model_b = models[:2]
    else:
        model_a, model_b = comparison_pair
    for model in (model_a, model_b):
        if model not in models:
            raise ValueError(f"Model {model!r} is absent from predictions")

    pair_df = predictions_df[predictions_df['model'].isin([model_a, model_b])].copy()
    key_cols = ['case_id', 'repeat', 'fold']
    for model in (model_a, model_b):
        model_df = pair_df[pair_df['model'] == model]
        duplicate_count = int(model_df.duplicated(key_cols, keep=False).sum())
        if duplicate_count:
            raise ValueError(f"{model} has {duplicate_count} duplicate case/repeat/fold rows")

    keys_a = set(map(tuple, pair_df[pair_df['model'] == model_a][key_cols].to_numpy()))
    keys_b = set(map(tuple, pair_df[pair_df['model'] == model_b][key_cols].to_numpy()))
    if keys_a != keys_b:
        only_a = len(keys_a - keys_b)
        only_b = len(keys_b - keys_a)
        raise ValueError(
            "Paired model coverage mismatch on (case_id, repeat, fold): "
            f"only_{model_a}={only_a}, only_{model_b}={only_b}"
        )

    outcome_a = pair_df[pair_df['model'] == model_a][key_cols + ['survival_months', 'event']]
    outcome_b = pair_df[pair_df['model'] == model_b][key_cols + ['survival_months', 'event']]
    outcome_check = outcome_a.merge(outcome_b, on=key_cols, suffixes=('_a', '_b'), validate='one_to_one')
    if not (
        np.allclose(outcome_check['survival_months_a'], outcome_check['survival_months_b'])
        and np.array_equal(outcome_check['event_a'].to_numpy(), outcome_check['event_b'].to_numpy())
    ):
        raise ValueError("Paired models do not share identical survival outcomes")

    repeats = sorted(int(r) for r in pd.unique(pair_df['repeat']))
    if not repeats:
        raise ValueError("No outer repeats found")

    case_ids_by_repeat = {
        r: set(pair_df[(pair_df['repeat'] == r) & (pair_df['model'] == model_a)]['case_id'])
        for r in repeats
    }
    reference_case_ids = case_ids_by_repeat[repeats[0]]
    if not reference_case_ids:
        raise ValueError("No patients found")
    if any(case_ids != reference_case_ids for case_ids in case_ids_by_repeat.values()):
        raise ValueError("Every repeat must contain the same patient cohort")
    case_ids = np.asarray(sorted(reference_case_ids), dtype=object)

    def score(frame: pd.DataFrame) -> float:
        if metric_func is None:
            return float(harrell_c_index(
                frame['survival_months'].to_numpy(dtype=float),
                frame['event'].to_numpy(dtype=bool),
                frame['risk_score'].to_numpy(dtype=float),
            ))
        try:
            return float(metric_func(
                frame['survival_months'].to_numpy(dtype=float),
                frame['event'].to_numpy(dtype=bool),
                frame['risk_score'].to_numpy(dtype=float),
            ))
        except TypeError:
            return float(metric_func(frame))

    observed_repeat_diffs = {}
    for r in repeats:
        repeat_df = pair_df[pair_df['repeat'] == r]
        observed_repeat_diffs[str(r)] = (
            score(repeat_df[repeat_df['model'] == model_a])
            - score(repeat_df[repeat_df['model'] == model_b])
        )

    rng = np.random.default_rng(seed)
    bootstrap_diffs = []
    invalid_iterations = 0
    for _ in range(n_iterations):
        sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        draws = pd.DataFrame({'case_id': sampled_ids, '_draw': np.arange(len(sampled_ids))})
        repeat_diffs = []
        try:
            for r in repeats:
                repeat_df = pair_df[pair_df['repeat'] == r]
                # Repeated sampled IDs deliberately create a many-to-many merge:
                # one row per draw and one row per model for that patient.
                sampled = draws.merge(repeat_df, on='case_id', how='left', validate='many_to_many')
                model_a_sample = sampled[sampled['model'] == model_a].sort_values('_draw')
                model_b_sample = sampled[sampled['model'] == model_b].sort_values('_draw')
                if len(model_a_sample) != len(case_ids) or len(model_b_sample) != len(case_ids):
                    raise ValueError("Bootstrap draw lost paired patient multiplicity")
                repeat_diffs.append(score(model_a_sample) - score(model_b_sample))
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
    nonpositive = int(np.sum(diffs <= 0))
    nonnegative = int(np.sum(diffs >= 0))
    p_value = min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1) / (len(diffs) + 1))

    return {
        'iterations': int(n_iterations),
        'iterations_requested': int(n_iterations),
        'iterations_valid': int(len(diffs)),
        'iterations_invalid': int(invalid_iterations),
        'status': 'SUCCESS',
        'metric': 'harrell_c_index' if metric_func is None else getattr(metric_func, '__name__', 'custom'),
        'model_a': model_a,
        'model_b': model_b,
        'n_patients': int(len(case_ids)),
        'n_repeats': int(len(repeats)),
        'repeats': repeats,
        'observed_repeat_differences': {
            key: float(value) for key, value in observed_repeat_diffs.items()
        },
        'observed_mean_difference': float(np.mean(list(observed_repeat_diffs.values()))),
        'mean_diff': float(np.mean(diffs)),
        'ci_lower': float(np.percentile(diffs, 2.5)),
        'ci_upper': float(np.percentile(diffs, 97.5)),
        'p_value': float(p_value),
        'fraction_better_a': float(np.mean(diffs > 0)),
        'repeat': repeat,
        'fold': fold,
        'methodology': 'patient_clustered_paired_bootstrap_repeat_mean',
        'multiplicity_preserved': True,
        'pairing_key': key_cols,
    }


def _compute_metric_cache(
    predictions_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric_func,
    default_metric
) -> Dict[Tuple, Dict]:
    """
    Pre-compute metrics for each (repeat, fold) combination.

    Returns a cache dict with keys (repeat, fold) and values containing
    metric values and case_ids for each model.
    """
    cache = {}

    for repeat in range(1, 6):
        for fold in range(1, 6):
            fold_pred = predictions_df[
                (predictions_df['repeat'] == repeat) &
                (predictions_df['fold'] == fold)
            ]

            model_a_pred = fold_pred[fold_pred['model'] == model_a]
            model_b_pred = fold_pred[fold_pred['model'] == model_b]

            cache_key = (repeat, fold)
            cache[cache_key] = {
                'case_ids': list(fold_pred['case_id'].unique()),
                f'model_{model_a}': None,
                f'model_{model_b}': None
            }

            try:
                if len(model_a_pred) > 0:
                    cache[cache_key][f'model_{model_a}'] = default_metric(model_a_pred) if metric_func is None else metric_func(model_a_pred)
                if len(model_b_pred) > 0:
                    cache[cache_key][f'model_{model_b}'] = default_metric(model_b_pred) if metric_func is None else metric_func(model_b_pred)
            except Exception:
                pass

    return cache


def aggregate_bootstrap_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate bootstrap results across multiple repeats and folds.

    Per Phase 3A reset: Proper aggregation for 5×5×5 nested CV.

    Parameters
    ----------
    results_list : list of dict
        List of bootstrap results, each from patient_level_paired_bootstrap

    Returns
    -------
    dict
        Aggregated statistics including per-fold breakdown and overall summary
    """
    if not results_list:
        return {'status': 'NO_RESULTS', 'message': 'Empty results list'}

    # Separate by repeat/fold
    by_repeat_fold = {}
    for r in results_list:
        key = (r.get('repeat'), r.get('fold'))
        if key not in by_repeat_fold:
            by_repeat_fold[key] = []
        by_repeat_fold[key].append(r)

    # Per-fold statistics
    fold_results = []
    for (repeat, fold), reps in sorted(by_repeat_fold.items()):
        if repeat is not None and fold is not None:
            # Average across any bootstrap runs for same repeat/fold
            p_values = [r['p_value'] for r in reps if r.get('p_value') is not None]
            mean_diffs = [r['mean_diff'] for r in reps if r.get('mean_diff') is not None]

            fold_results.append({
                'repeat': repeat,
                'fold': fold,
                'mean_p_value': np.mean(p_values) if p_values else None,
                'mean_diff': np.mean(mean_diffs) if mean_diffs else None,
                'n_comparisons': len(reps)
            })

    # Extract comparable pairs across all results
    # Group by (model_a, model_b)
    by_pair = {}
    for r in results_list:
        pair_key = (r.get('model_a'), r.get('model_b'))
        if pair_key not in by_pair:
            by_pair[pair_key] = []
        by_pair[pair_key].append(r)

    pair_summaries = []
    for (model_a, model_b), pair_results in by_pair.items():
        p_values = [r['p_value'] for r in pair_results if r.get('p_value') is not None]
        mean_diffs = [r['mean_diff'] for r in pair_results if r.get('mean_diff') is not None]
        ci_lowers = [r['ci_lower'] for r in pair_results if r.get('ci_lower') is not None]
        ci_uppers = [r['ci_upper'] for r in pair_results if r.get('ci_upper') is not None]

        # Fisher's method for combining p-values
        if p_values:
            from scipy import stats
            # Handle p-values of 0 (very significant)
            safe_p = [max(p, 1e-10) for p in p_values]
            combined_stat = -2 * sum(np.log(safe_p))
            combined_df = 2 * len(safe_p)
            combined_p = 1 - stats.chi2.cdf(combined_stat, combined_df)
        else:
            combined_p = None

        # Mean CI across folds (approximation)
        mean_ci_lower = np.mean(ci_lowers) if ci_lowers else None
        mean_ci_upper = np.mean(ci_uppers) if ci_uppers else None

        pair_summaries.append({
            'model_a': model_a,
            'model_b': model_b,
            'n_folds': len(pair_results),
            'combined_p_value': float(combined_p) if combined_p is not None else None,
            'mean_diff': float(np.mean(mean_diffs)) if mean_diffs else None,
            'std_diff': float(np.std(mean_diffs)) if mean_diffs else None,
            'mean_ci_lower': float(mean_ci_lower) if mean_ci_lower is not None else None,
            'mean_ci_upper': float(mean_ci_upper) if mean_ci_upper is not None else None,
            'folds_better_a': sum(1 for d in mean_diffs if d is not None and d > 0),
            'fraction_better_a': sum(1 for d in mean_diffs if d is not None and d > 0) / len(mean_diffs) if mean_diffs else 0
        })

    return {
        'status': 'AGGREGATED',
        'n_results': len(results_list),
        'n_unique_folds': len(fold_results),
        'fold_summary': fold_results,
        'pairwise_comparisons': pair_summaries,
        'model_pairs_compared': list(by_pair.keys())
    }


def run_full_bootstrap_comparison(
    predictions_df: pd.DataFrame,
    model_pairs: List[Tuple[str, str]],
    n_iterations: int = 1000,
    seed: int = 456,
    repeat: Optional[int] = None,
    fold: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run bootstrap comparison for multiple model pairs.

    Per Phase 3A reset: Compares all specified model pairs with proper
    repeat/fold tracking for aggregation.

    Parameters
    ----------
    predictions_df : DataFrame
        Predictions with columns: case_id, model, risk_score, survival_months, event
    model_pairs : list of tuple
        List of (model_a, model_b) tuples to compare
    n_iterations : int
        Number of bootstrap iterations per comparison
    seed : int
        Random seed (incremented for each comparison)
    repeat : int, optional
        Outer repeat number
    fold : int, optional
        Outer fold number

    Returns
    -------
    dict
        Results for all comparisons with metadata
    """
    results = []

    for i, (model_a, model_b) in enumerate(model_pairs):
        comparison_seed = seed + i  # Different seed per comparison for independence
        result = patient_level_paired_bootstrap(
            predictions_df=predictions_df,
            n_iterations=n_iterations,
            seed=comparison_seed,
            comparison_pair=(model_a, model_b),
            repeat=repeat,
            fold=fold
        )
        results.append(result)

    # Aggregate if multiple comparisons
    if len(results) > 1:
        aggregated = aggregate_bootstrap_results(results)
        return {
            'individual_results': results,
            'aggregated': aggregated,
            'metadata': {
                'n_iterations': n_iterations,
                'repeat': repeat,
                'fold': fold,
                'n_comparisons': len(model_pairs)
            }
        }
    else:
        return {
            'result': results[0] if results else None,
            'metadata': {
                'n_iterations': n_iterations,
                'repeat': repeat,
                'fold': fold
            }
        }
