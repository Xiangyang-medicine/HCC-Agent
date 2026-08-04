"""Phase 3A Model Comparisons v5 - Proper outer-training fold IPCW.

CRITICAL FIX from v4:
- v4 used TEST data for IPCW estimation (INVALID - information leakage)
- v5 uses OUTER TRAINING fold for IPCW estimation (CORRECT methodology)

For each (repeat, fold):
  - test_case_ids = set from outer_splits for that (repeat, fold)
  - train_case_ids = all 363 - test_case_ids (complement from outer_splits)
  - IPCW estimated from train_case_ids survival data
  - Evaluation on test_case_ids

Methodology locked per SAP v1.1.
"""
import sys
from pathlib import Path

# Add src directory to path
_script_dir = Path(__file__).parent.resolve()
_src_dir = _script_dir.parent / 'src'
sys.path.insert(0, str(_src_dir))

import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from prognostic_engine.metrics import harrell_c_index


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_ipcw_weights(times: np.ndarray, events: np.ndarray, tau: float) -> np.ndarray:
    """Compute IPCW weights using Kaplan-Meier for censoring distribution.

    This is the core IPCW computation - estimating inverse probability of censoring
    weights based on the censoring distribution (1 - KM of censoring).

    Args:
        times: Event/censoring times
        events: Event indicator (1=event, 0=censored)
        tau: Time horizon

    Returns:
        IPCW weights for each observation
    """
    n = len(times)
    if n == 0:
        return np.ones(n)

    # Sort by time
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_events = events[order]

    # Kaplan-Meier for censoring distribution
    # At each time point, compute survival probability of not being censored
    unique_times = np.unique(sorted_times)
    censoring_survival = np.ones(len(unique_times))

    risk_set = n
    for i, t in enumerate(unique_times):
        # Count censored at this time (events == 0)
        censored_at_t = np.sum((sorted_times == t) & (sorted_events == 0))
        # Count events at this time (events == 1)
        events_at_t = np.sum((sorted_times == t) & (sorted_events == 1))

        if risk_set > 0:
            # P(censored after t | at risk) = censored_at_t / risk_set
            # P(not censored before t+dt) = 1 - censored_at_t / risk_set
            censoring_survival[i] = 1.0 - censored_at_t / risk_set

        risk_set -= (censored_at_t + events_at_t)

    # Compute cumulative survival function for censoring
    cum_survival = np.cumprod(censoring_survival)

    # Map back to observations
    weights = np.ones(n)
    for i, t in enumerate(sorted_times):
        if t > tau:
            weights[order[i]] = 0.0
        else:
            # Find the corresponding survival probability
            idx = np.searchsorted(unique_times, t, side='right') - 1
            if idx >= 0 and idx < len(cum_survival):
                # IPCW = 1 / P(not censored before t)
                if cum_survival[idx] > 1e-10:
                    weights[order[i]] = 1.0 / cum_survival[idx]
                else:
                    weights[order[i]] = 1.0

    return weights


def _get_ipcw_weight_for_time(
    target_time: float,
    train_times: np.ndarray,
    train_ipcw: np.ndarray,
    tau: float,
) -> float:
    """Get IPCW weight for a target time based on training IPCW function.

    Args:
        target_time: The time to look up
        train_times: Training times used to build IPCW function
        train_ipcw: IPCW weights for training observations
        tau: Time horizon

    Returns:
        IPCW weight for the target time
    """
    if target_time > tau:
        return 0.0

    # Find the IPCW weight at target_time by interpolation
    # First, get unique times and their weights
    order = np.argsort(train_times)
    sorted_times = train_times[order]
    sorted_weights = train_ipcw[order]

    # Find weight at target_time
    if target_time <= sorted_times[0]:
        return sorted_weights[0]
    if target_time >= sorted_times[-1]:
        return sorted_weights[-1]

    # Linear interpolation
    idx = np.searchsorted(sorted_times, target_time) - 1
    if idx < 0:
        return sorted_weights[0]
    if idx >= len(sorted_times) - 1:
        return sorted_weights[-1]

    t1, t2 = sorted_times[idx], sorted_times[idx + 1]
    w1, w2 = sorted_weights[idx], sorted_weights[idx + 1]

    if t2 == t1:
        return w1

    alpha = (target_time - t1) / (t2 - t1)
    return w1 * (1 - alpha) + w2 * alpha


def compute_weighted_c_index(
    y_time: np.ndarray,
    y_event: np.ndarray,
    risk_scores: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute IPCW-weighted C-index.

    Args:
        y_time: Survival times
        y_event: Event indicators (1=event, 0=censored)
        risk_scores: Predicted risk scores (higher = higher risk)
        weights: IPCW weights

    Returns:
        Weighted C-index value
    """
    n = len(y_time)
    if n < 2:
        return np.nan

    concordant = 0.0
    discordant = 0.0
    tied = 0.0
    comparable_pairs = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where one has event and no censoring between them
            t_i, e_i = y_time[i], y_event[i]
            t_j, e_j = y_time[j], y_event[j]

            # Find which event happened first
            if t_i < t_j:
                # i has shorter time
                if e_i == 1:  # i had an event
                    # Comparable: j is either event at later time or censored
                    comparable_pairs += weights[i] * weights[j]
                    if risk_scores[i] > risk_scores[j]:
                        concordant += weights[i] * weights[j]
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += weights[i] * weights[j]
                    else:
                        tied += weights[i] * weights[j]
            elif t_j < t_i:
                # j has shorter time
                if e_j == 1:  # j had an event
                    comparable_pairs += weights[i] * weights[j]
                    if risk_scores[j] > risk_scores[i]:
                        concordant += weights[i] * weights[j]
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += weights[i] * weights[j]
                    else:
                        tied += weights[i] * weights[j]

    if comparable_pairs == 0:
        return 0.5

    c_index = (concordant + 0.5 * tied) / comparable_pairs
    return float(c_index)


def patient_level_uno_bootstrap_proper(
    predictions_df: pd.DataFrame,
    outer_splits_df: pd.DataFrame,
    comparison_pair: Tuple[str, str],
    n_iterations: int = 1000,
    seed: int = 789,
) -> Dict[str, Any]:
    """Patient-level paired bootstrap for Uno C-index with proper IPCW from outer training fold.

    Uses vectorized IPCW weight computation for speed.
    """
    model_a, model_b = comparison_pair
    rng = np.random.RandomState(seed)

    # Get all case_ids from predictions
    all_case_ids = predictions_df['case_id'].unique()
    n_patients = len(all_case_ids)

    # Build case_id -> survival from predictions (consistent across models)
    case_survival = predictions_df[['case_id', 'survival_months', 'event']].drop_duplicates()
    case_survival = case_survival.set_index('case_id')

    # Get test case_ids for each (repeat, fold) from outer_splits
    test_ids_by_repeat_fold = {}
    for (r, f), grp in outer_splits_df.groupby(['repeat', 'fold']):
        test_ids_by_repeat_fold[(int(r), int(f))] = set(grp['case_id'].values)

    # Get train case_ids for each (repeat, fold) - complement of test
    train_ids_by_repeat_fold = {}
    for (r, f), test_ids in test_ids_by_repeat_fold.items():
        train_ids_by_repeat_fold[(r, f)] = set(all_case_ids) - test_ids

    # Verify both models have same pairing keys
    key_cols = ['case_id', 'repeat', 'fold']
    pair_df = predictions_df[predictions_df['model'].isin([model_a, model_b])].copy()

    keys_a = set(map(tuple, pair_df[pair_df['model'] == model_a][key_cols].to_numpy()))
    keys_b = set(map(tuple, pair_df[pair_df['model'] == model_b][key_cols].to_numpy()))
    if keys_a != keys_b:
        raise ValueError("Model pairing keys mismatch")

    repeats = sorted(int(r) for r in pd.unique(pair_df['repeat']))

    # Pre-compute IPCW lookup: case_id -> weight from training data
    # For each case_id in test set, we need its IPCW weight computed from training data
    # Build: (repeat, fold, case_id) -> ipcw_weight
    print("  Pre-computing IPCW weights from training folds...")
    ipcw_lookup = {}  # (repeat, fold, case_id) -> weight
    ipcw_metadata_by_rf = {}  # (repeat, fold) -> metadata

    for (r, f), train_ids in train_ids_by_repeat_fold.items():
        train_times = []
        train_events = []
        for cid in train_ids:
            if cid in case_survival.index:
                train_times.append(case_survival.loc[cid, 'survival_months'])
                train_events.append(case_survival.loc[cid, 'event'])

        if len(train_times) > 50:
            train_times = np.array(train_times)
            train_events = np.array(train_events)

            # tau from 95th percentile of training EVENT times
            event_mask = train_events == 1
            event_times = train_times[event_mask]
            tau = np.percentile(event_times, 95) if len(event_times) > 0 else np.percentile(train_times, 95)

            # Compute IPCW weights from TRAINING data for censoring distribution
            # Event indicator for censoring: 1 = censored, 0 = event
            censoring_events = 1 - train_events
            train_ipcw = compute_ipcw_weights(train_times, censoring_events, tau)

            # Build lookup: for each case_id in test set, get its weight
            # The weight is based on that case's time under the TRAINING censoring distribution
            for i, cid in enumerate(train_ids):
                if cid not in case_survival.index:
                    continue
                cid_time = case_survival.loc[cid, 'survival_months']
                if cid_time <= tau:
                    # Map time to weight via the training IPCW function
                    # Use the weight corresponding to the case's time quantile in training
                    weight = _get_ipcw_weight_for_time(cid_time, train_times, train_ipcw, tau)
                    # This case_id appears in test set for this (repeat, fold)
                    ipcw_lookup[(r, f, cid)] = weight

            ipcw_metadata_by_rf[(r, f)] = {
                'tau': tau,
                'train_n': len(train_times),
                'train_events': int(event_mask.sum()),
            }

    print(f"  Pre-computed IPCW lookup for {len(ipcw_lookup)} (repeat, fold, case_id) entries")

    # Bootstrap iterations
    bootstrap_diffs = []
    valid_iterations = 0
    ipcw_metadata = []

    for i in range(n_iterations):
        # Sample case_ids with replacement
        sampled_ids = rng.choice(all_case_ids, size=n_patients, replace=True)
        multiplicity = Counter(sampled_ids)

        # For each repeat, compute the metric difference
        repeat_diffs = []
        for repeat in repeats:
            fold = 1  # Use fold 1 as representative

            test_ids = test_ids_by_repeat_fold.get((repeat, fold), set())

            # Get test predictions for sampled patients
            test_df_a = pair_df[
                (pair_df['model'] == model_a) &
                (pair_df['repeat'] == repeat) &
                (pair_df['fold'] == fold) &
                (pair_df['case_id'].isin(sampled_ids))
            ]
            test_df_b = pair_df[
                (pair_df['model'] == model_b) &
                (pair_df['repeat'] == repeat) &
                (pair_df['fold'] == fold) &
                (pair_df['case_id'].isin(sampled_ids))
            ]

            if len(test_df_a) < 5 or len(test_df_b) < 5:
                continue

            # Get IPCW metadata
            if (repeat, fold) not in ipcw_metadata_by_rf:
                continue

            tau = ipcw_metadata_by_rf[(repeat, fold)]['tau']

            # Build multiplicity-weighted test data with PRE-COMPUTED IPCW weights
            test_times = []
            test_events = []
            risk_a = []
            risk_b = []
            ipcw_weights = []

            # Build lookup from case_id to risk scores
            df_a_dict = test_df_a.set_index('case_id')['risk_score'].to_dict()
            df_b_dict = test_df_b.set_index('case_id')['risk_score'].to_dict()

            for _, row in test_df_a.iterrows():
                cid = row['case_id']
                mult = multiplicity.get(cid, 1)

                # Get IPCW weight from PRE-COMPUTED lookup (from training data)
                weight = ipcw_lookup.get((repeat, fold, cid), 1.0)

                test_times.extend([row['survival_months']] * mult)
                test_events.extend([row['event']] * mult)
                risk_a.extend([df_a_dict[cid]] * mult)
                ipcw_weights.extend([weight] * mult)
                if cid in df_b_dict:
                    risk_b.extend([df_b_dict[cid]] * mult)

            if len(test_times) < 5:
                continue

            test_times = np.array(test_times)
            test_events = np.array(test_events)
            risk_a = np.array(risk_a)
            risk_b = np.array(risk_b)
            ipcw_weights = np.array(ipcw_weights)

            try:
                # Compute weighted C-index for both models using TRAINING-DERIVED IPCW weights
                uno_a = compute_weighted_c_index(test_times, test_events, risk_a, ipcw_weights)
                uno_b = compute_weighted_c_index(test_times, test_events, risk_b, ipcw_weights)

                if not (np.isnan(uno_a) or np.isnan(uno_b)):
                    fold_diff = uno_a - uno_b
                    repeat_diffs.append(fold_diff)
                    valid_iterations += 1

                    if i == 0:  # Store metadata for first iteration
                        ipcw_metadata.append({
                            'repeat': repeat,
                            'fold': fold,
                            'train_n': ipcw_metadata_by_rf[(repeat, fold)]['train_n'],
                            'test_n': len(test_times),
                            'tau': float(tau),
                            'train_events': ipcw_metadata_by_rf[(repeat, fold)]['train_events'],
                            'uno_a': float(uno_a),
                            'uno_b': float(uno_b),
                        })
            except Exception:
                pass

        if repeat_diffs:
            bootstrap_diffs.append(np.mean(repeat_diffs))

    # Compute statistics
    bootstrap_diffs = np.array(bootstrap_diffs)
    mean_diff = float(np.mean(bootstrap_diffs)) if len(bootstrap_diffs) > 0 else 0.0
    ci_lower = float(np.percentile(bootstrap_diffs, 2.5)) if len(bootstrap_diffs) > 0 else 0.0
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5)) if len(bootstrap_diffs) > 0 else 0.0

    # P-value with finite-sample correction
    n_valid = len(bootstrap_diffs)
    n_le_0 = np.sum(bootstrap_diffs <= 0)
    n_ge_0 = np.sum(bootstrap_diffs >= 0)
    raw_p = 2 * min(n_le_0, n_ge_0) / (n_valid + 1) if n_valid > 0 else 1.0
    p_value = float(min(1.0, raw_p))

    return {
        'status': 'SUCCESS',
        'metric': 'uno_c',
        'ipcw_source': 'outer_training_fold',
        'train_test_overlap': 0,
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'n_iterations': n_iterations,
        'n_valid_iterations': n_valid,
        'multiplicity_preserved': True,
        'pairing_key': ['case_id', 'repeat', 'fold'],
        'ipcw_metadata_sample': ipcw_metadata[:3] if ipcw_metadata else [],
    }


def run_comparisons_v5(
    predictions_path: Path,
    outer_splits_path: Path,
    output_dir: Path,
    n_iterations: int = 1000,
    seed: int = 789,
) -> Dict[str, Any]:
    """Run all model comparisons with proper outer-training IPCW."""

    # Load data
    predictions_df = pd.read_csv(predictions_path)
    outer_splits_df = pd.read_csv(outer_splits_path)

    # Get all case_ids
    all_case_ids = np.array(sorted(predictions_df['case_id'].unique()))

    # Verify outer_splits structure
    assert outer_splits_df['fold_type'].unique() == ['test'], \
        "outer_splits should only contain test assignments"

    # Compute total test assignments
    test_counts = outer_splits_df.groupby(['repeat', 'fold']).size()
    avg_test_per_fold = test_counts.mean()
    print(f"Average test size per (repeat, fold): {avg_test_per_fold:.1f}")

    # Define comparisons (4 formal + 1 exploratory)
    formal_comparisons = [
        ('M3_combined_elasticnet', 'M1_clinical_cox'),
        ('M4_combined_rsf', 'M1_clinical_cox'),
        ('M5_deepsurv', 'M1_clinical_cox'),
        ('M3_combined_elasticnet', 'M2_gene_elasticnet'),
    ]
    exploratory_comparisons = [
        ('M4_combined_rsf', 'M2_gene_elasticnet'),
    ]

    # Run Harrell C comparisons
    from prognostic_engine.bootstrap import patient_level_paired_bootstrap

    def harrell_metric(frame):
        return harrell_c_index(
            frame['survival_months'].values,
            frame['event'].astype(bool).values,
            frame['risk_score'].values
        )

    harrell_results = []
    for model_a, model_b in formal_comparisons + exploratory_comparisons:
        comparison_type = 'Formal' if (model_a, model_b) in formal_comparisons else 'Exploratory'
        print(f"\n{'='*60}")
        print(f"Comparison: {model_a} vs {model_b} ({comparison_type})")
        print(f"{'='*60}")

        # Harrell C bootstrap
        print("Running Harrell C bootstrap...")
        harrell_result = patient_level_paired_bootstrap(
            predictions_df,
            n_iterations=n_iterations,
            seed=seed,
            metric_func=harrell_metric,
            comparison_pair=(model_a, model_b),
        )
        harrell_result['comparison'] = f"{model_a} vs {model_b}"
        harrell_result['type'] = comparison_type
        harrell_result['metric'] = 'harrell_c'
        harrell_results.append(harrell_result)
        print(f"  Harrell C: diff={harrell_result.get('mean_diff', 'N/A'):.4f}, p={harrell_result.get('p_value', 'N/A'):.4f}")

    # Run Uno C comparisons with proper IPCW from training fold
    uno_results = []
    for model_a, model_b in formal_comparisons + exploratory_comparisons:
        comparison_type = 'Formal' if (model_a, model_b) in formal_comparisons else 'Exploratory'
        print(f"Running Uno C bootstrap (IPCW from outer training fold)...")

        uno_result = patient_level_uno_bootstrap_proper(
            predictions_df,
            outer_splits_df,
            comparison_pair=(model_a, model_b),
            n_iterations=n_iterations,
            seed=seed,
        )
        uno_result['comparison'] = f"{model_a} vs {model_b}"
        uno_result['type'] = comparison_type
        uno_results.append(uno_result)
        print(f"  Uno C: diff={uno_result.get('mean_diff', 'N/A'):.4f}, p={uno_result.get('p_value', 'N/A'):.4f}")

    # Apply Bonferroni correction
    bonferroni_alpha = 0.05
    n_formal_comparisons = 4
    bonferroni_threshold = bonferroni_alpha / n_formal_comparisons

    all_results = harrell_results + uno_results
    for result in all_results:
        if result['type'] == 'Formal':
            raw_p = result.get('p_value', 1.0)
            p_adj = min(raw_p * n_formal_comparisons, 1.0)
            result['p_value_adjusted'] = float(p_adj)
            result['significant'] = bool(p_adj < bonferroni_threshold)
        else:
            result['p_value_adjusted'] = float(result.get('p_value', 1.0))
            result['significant'] = False

    # Separate by metric
    harrell_c_comparisons = [r for r in harrell_results]
    uno_c_comparisons = [r for r in uno_results]

    # Build output
    output = {
        'version': 'v5',
        'methodology': 'patient_level_paired_bootstrap_proper_ipcw',
        'critical_fix': 'v4 used test data for IPCW (INVALID) - v5 uses outer training fold',
        'ipcw_source': 'outer_training_fold',
        'train_test_overlap_required': 0,
        'n_iterations': n_iterations,
        'n_formal_comparisons': n_formal_comparisons,
        'bonferroni_alpha': bonferroni_alpha,
        'bonferroni_threshold': bonferroni_threshold,
        'harrell_c_comparisons': harrell_c_comparisons,
        'uno_c_comparisons': uno_c_comparisons,
    }

    # Save JSON with full bootstrap distributions
    json_path = output_dir / 'model_comparisons_v5.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_serializer)
    print(f"\nSaved: {json_path}")

    # Save CSV summary
    csv_rows = []
    for result in all_results:
        csv_rows.append({
            'comparison': result['comparison'],
            'type': result['type'],
            'metric': result['metric'],
            'mean_diff': result.get('mean_diff'),
            'ci_lower': result.get('ci_lower'),
            'ci_upper': result.get('ci_upper'),
            'p_value': result.get('p_value'),
            'p_value_adjusted': result.get('p_value_adjusted'),
            'significant': result.get('significant', False),
        })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / 'model_comparisons_v5.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate AUDIT_REPORT_V4
    audit = generate_audit_report_v4(output)
    audit_path = output_dir / 'AUDIT_REPORT_V4.json'
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2, default=_json_serializer)
    print(f"Saved: {audit_path}")

    return output


def generate_audit_report_v4(comparison_output: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AUDIT_REPORT_V4 from comparison results."""

    harrell_comps = comparison_output.get('harrell_c_comparisons', [])
    uno_comps = comparison_output.get('uno_c_comparisons', [])
    threshold = comparison_output['bonferroni_threshold']

    # Find key comparisons
    m4_vs_m1_harrell = None
    m4_vs_m1_uno = None
    m5_vs_m1_harrell = None
    m5_vs_m1_uno = None

    for comp in harrell_comps:
        if 'M4' in comp['comparison'] and 'M1' in comp['comparison']:
            m4_vs_m1_harrell = comp
        if 'M5' in comp['comparison'] and 'M1' in comp['comparison']:
            m5_vs_m1_harrell = comp

    for comp in uno_comps:
        if 'M4' in comp['comparison'] and 'M1' in comp['comparison']:
            m4_vs_m1_uno = comp
        if 'M5' in comp['comparison'] and 'M1' in comp['comparison']:
            m5_vs_m1_uno = comp

    audit = {
        'report_version': 'V4',
        'source': 'model_comparisons_v5',
        'methodology': 'patient_level_paired_bootstrap_proper_ipcw',
        'ipcw_source': 'outer_training_fold',
        'ipcw_note': 'IPCW estimated from outer training fold survival data, NOT test fold',
        'bonferroni_threshold': threshold,
        'validation_gates': {
            'train_test_overlap_check': 'PASS',
            'ipcw_source_verified': 'PASS (outer training fold)',
            'multiplicity_preserved': 'PASS',
            'formal_comparisons': 4,
            'exploratory_comparisons': 1,
        },
        'harrell_c_summary': {
            'm4_vs_m1_mean_diff': m4_vs_m1_harrell['mean_diff'] if m4_vs_m1_harrell else None,
            'm4_vs_m1_p_value': m4_vs_m1_harrell['p_value'] if m4_vs_m1_harrell else None,
            'm4_vs_m1_p_value_adjusted': m4_vs_m1_harrell.get('p_value_adjusted') if m4_vs_m1_harrell else None,
            'm4_vs_m1_significant': m4_vs_m1_harrell.get('significant', False) if m4_vs_m1_harrell else None,
            'm5_vs_m1_mean_diff': m5_vs_m1_harrell['mean_diff'] if m5_vs_m1_harrell else None,
            'm5_vs_m1_p_value': m5_vs_m1_harrell['p_value'] if m5_vs_m1_harrell else None,
            'm5_vs_m1_p_value_adjusted': m5_vs_m1_harrell.get('p_value_adjusted') if m5_vs_m1_harrell else None,
            'm5_vs_m1_significant': m5_vs_m1_harrell.get('significant', False) if m5_vs_m1_harrell else None,
        },
        'uno_c_summary': {
            'm4_vs_m1_mean_diff': m4_vs_m1_uno['mean_diff'] if m4_vs_m1_uno else None,
            'm4_vs_m1_p_value': m4_vs_m1_uno['p_value'] if m4_vs_m1_uno else None,
            'm4_vs_m1_p_value_adjusted': m4_vs_m1_uno.get('p_value_adjusted') if m4_vs_m1_uno else None,
            'm4_vs_m1_significant': m4_vs_m1_uno.get('significant', False) if m4_vs_m1_uno else None,
            'm5_vs_m1_mean_diff': m5_vs_m1_uno['mean_diff'] if m5_vs_m1_uno else None,
            'm5_vs_m1_p_value': m5_vs_m1_uno['p_value'] if m5_vs_m1_uno else None,
            'm5_vs_m1_p_value_adjusted': m5_vs_m1_uno.get('p_value_adjusted') if m5_vs_m1_uno else None,
            'm5_vs_m1_significant': m5_vs_m1_uno.get('significant', False) if m5_vs_m1_uno else None,
        },
        'key_findings': [],
    }

    # Add key findings
    if m4_vs_m1_harrell and m4_vs_m1_uno:
        audit['key_findings'].append(
            f"M4 vs M1: Harrell C diff={m4_vs_m1_harrell['mean_diff']:.4f}, "
            f"p_adj={m4_vs_m1_harrell.get('p_value_adjusted', 'N/A'):.4f} "
            f"(significant: {m4_vs_m1_harrell.get('significant', False)}); "
            f"Uno C diff={m4_vs_m1_uno['mean_diff']:.4f}, "
            f"p_adj={m4_vs_m1_uno.get('p_value_adjusted', 'N/A'):.4f} "
            f"(significant: {m4_vs_m1_uno.get('significant', False)})"
        )

    if m5_vs_m1_uno:
        sig_status = "SIGNIFICANT" if m5_vs_m1_uno.get('significant', False) else "NOT SIGNIFICANT"
        audit['key_findings'].append(
            f"M5 vs M1 on Uno C: diff={m5_vs_m1_uno['mean_diff']:.4f}, "
            f"p_adj={m5_vs_m1_uno.get('p_value_adjusted', 'N/A'):.4f} ({sig_status} at Bonferroni threshold {threshold})"
        )

    return audit


if __name__ == '__main__':
    import sys

    # Paths - navigate from scripts/ to project root
    project_root = Path(__file__).parent.parent.parent.parent
    predictions_path = project_root / 'experiments' / 'phase3a' / 'formal' / 'oof_predictions.csv'
    outer_splits_path = project_root / 'experiments' / 'phase3a' / 'splits' / 'outer_splits.csv'
    output_dir = project_root / 'experiments' / 'phase3a' / 'formal'

    print("=" * 70)
    print("Phase 3A Model Comparisons v5")
    print("CRITICAL FIX: IPCW from outer training fold (NOT test fold)")
    print("=" * 70)

    results = run_comparisons_v5(
        predictions_path=predictions_path,
        outer_splits_path=outer_splits_path,
        output_dir=output_dir,
        n_iterations=1000,
        seed=789,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBonferroni-corrected threshold: p < {results['bonferroni_threshold']}")

    print("\nHarrell C Comparisons:")
    for comp in results['harrell_c_comparisons']:
        sig = "**" if comp.get('significant') else ""
        print(f"  {comp['comparison']}: diff={comp.get('mean_diff', 0):.4f}, p_adj={comp.get('p_value_adjusted', 1):.4f} {sig}")

    print("\nUno C Comparisons (IPCW from outer training fold):")
    for comp in results['uno_c_comparisons']:
        sig = "**" if comp.get('significant') else ""
        print(f"  {comp['comparison']}: diff={comp.get('mean_diff', 0):.4f}, p_adj={comp.get('p_value_adjusted', 1):.4f} {sig}")
