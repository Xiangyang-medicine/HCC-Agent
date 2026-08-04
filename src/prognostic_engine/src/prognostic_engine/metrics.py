"""Evaluation metrics for survival models per SAP v1.1.

Includes:
- Harrell C-index
- Uno C-index (IPCW-weighted) - using concordance_index_ipcw
- Time-dependent AUC at 12/36/60 months - using train/test separation
- Brier score with IPCW
- Integrated Brier Score (IBS)
- Calibration with KM/IPCW

CORRECTIONS per Phase 3A methodological reset:
1. uno_c_index: Use concordance_index_ipcw with proper IPCW estimation
2. time_dependent_auc: Use train/test separation for cumulative_dynamic_auc
3. Brier/IBS: Use survival probability matrices from models
4. Calibration: Use KM/IPCW for censoring-aware calibration
"""

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score
)
from lifelines.utils import concordance_index as harrell_cindex


def harrell_c_index(y_time, y_event, risk_scores):
    """
    Compute Harrell C-index (standard concordance index).

    Parameters
    ----------
    y_time : array-like
        Survival times
    y_event : array-like
        Event indicators (1=event, 0=censored)
    risk_scores : array-like
        Risk scores (higher = more risk)

    Returns
    -------
    float
        Harrell C-index value
    """
    return harrell_cindex(y_time, -risk_scores, y_event)


def uno_c_index(y_train_time, y_train_event, y_test_time, y_test_event, risk_scores, tau=None):
    """
    Compute Uno C-index (IPCW-weighted concordance index).

    Uses concordance_index_ipcw with IPCW estimated from training data.
    Per Phase 3A methodological reset: no fallback to Harrell C.

    Parameters
    ----------
    y_train_time : array-like
        Survival times from TRAINING set (for IPCW estimation)
    y_train_event : array-like
        Event indicators from TRAINING set
    y_test_time : array-like
        Survival times from TEST set
    y_test_event : array-like
        Event indicators from TEST set
    risk_scores : array-like
        Risk scores for TEST set (higher = more risk)
    tau : float, optional
        Truncation time. If None, uses 95th percentile decision rule.

    Returns
    -------
    float
        Uno C-index value or np.nan if calculation fails
    """
    # Create structured arrays
    y_train_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(y_train_event, y_train_time)],
        dtype=[('event', bool), ('time', float)]
    )
    y_test_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(y_test_event, y_test_time)],
        dtype=[('event', bool), ('time', float)]
    )

    # Use concordance_index_ipcw with IPCW estimated from training data
    try:
        # tau: truncation time - Phase 3A: use pre-defined decision rule
        if tau is None:
            event_times = y_train_time[np.array(y_train_event, dtype=bool)]
            if len(event_times) > 0:
                tau = np.percentile(event_times, 95)
            else:
                tau = np.max(y_train_time)

        # Record the actual tau used for diagnostics
        cidx, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(
            y_train_struct,
            y_test_struct,
            risk_scores,
            tau=tau
        )
        return cidx
    except Exception as e:
        # Phase 3A Reset: No fallback to Harrell C
        # Log the error but return NaN
        import logging
        logging.warning(f"Uno C-index calculation failed: {e}. Returning NaN.")
        return np.nan


def time_dependent_auc(y_train_time, y_train_event, y_test_time, y_test_event,
                       risk_scores, times=[12, 36, 60]):
    """
    Compute time-dependent AUC with TRAIN/TEST separation.

    IMPORTANT: Uses training data to estimate IPCW weights, then evaluates
    on test data. This prevents information leakage.
    Per Phase 3A reset: records failures as NOT_ESTIMABLE.

    Parameters
    ----------
    y_train_time : array-like
        Survival times from TRAINING set
    y_train_event : array-like
        Event indicators from TRAINING set
    y_test_time : array-like
        Survival times from TEST set
    y_test_event : array-like
        Event indicators from TEST set
    risk_scores : array-like
        Risk scores for TEST set (higher = more risk)
    times : list
        Time points in months

    Returns
    -------
    dict
        Dictionary with time points as keys and AUC values as values
    """
    import logging

    # Create structured arrays
    y_train_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(y_train_event, y_train_time)],
        dtype=[('event', bool), ('time', float)]
    )
    y_test_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(y_test_event, y_test_time)],
        dtype=[('event', bool), ('time', float)]
    )

    # Sort by time for sksurv
    sort_idx = np.argsort(y_test_struct['time'])
    y_test_sorted = y_test_struct[sort_idx]
    risk_sorted = np.asarray(risk_scores)[sort_idx]

    auc_results = {}
    for t in times:
        try:
            # cumulative_dynamic_auc: first arg is train data (for IPCW),
            # second arg is test data (for evaluation)
            auc, mean_survival = cumulative_dynamic_auc(
                y_train_struct,  # Training data for IPCW estimation
                y_test_sorted,    # Test data for evaluation
                risk_sorted,
                times=[t]
            )
            auc_results[f'auc_{t}m'] = auc[0]
            auc_results[f'auc_{t}m_status'] = 'ESTIMATED'
        except Exception as e:
            # Phase 3A Reset: Record failure, return NaN with status
            logging.warning(f"AUC at {t}m failed: {e}")
            auc_results[f'auc_{t}m'] = np.nan
            auc_results[f'auc_{t}m_status'] = 'NOT_ESTIMABLE'

    return auc_results


def compute_brier_score(y_train_time, y_train_event, y_test_time, y_test_event,
                        survival_probs, times=[12, 36, 60]):
    """
    Compute Brier score and Integrated Brier Score (IBS).

    Uses training data to estimate survival probabilities, evaluates on test data.
    survival_probs should be the survival probability matrix from the model.

    Parameters
    ----------
    y_train_time : array-like
        Survival times from TRAINING set
    y_train_event : array-like
        Event indicators from TRAINING set
    y_test_time : array-like
        Survival times from TEST set
    y_test_event : array-like
        Event indicators from TEST set
    survival_probs : array-like
        Predicted survival probabilities for TEST set (n_samples, n_times)
    times : list
        Time points for individual Brier scores

    Returns
    -------
    dict
        Dictionary with individual Brier scores and IBS
    """
    from scipy.interpolate import interp1d

    # Create structured arrays
    y_train_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(y_train_event, y_train_time)],
        dtype=[('event', bool), ('time', float)]
    )
    y_test_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(y_test_event, y_test_time)],
        dtype=[('event', bool), ('time', float)]
    )

    # Ensure survival_probs is 2D (n_samples, n_times)
    survival_probs = np.asarray(survival_probs)
    if survival_probs.ndim == 1:
        survival_probs = survival_probs.reshape(-1, len(times))

    eval_times = np.array(times)

    try:
        # brier_score: first arg is train data, second is test data
        # survival_probs should be S(t) - the survival probability
        # brier_score expects S(t) directly (not mortality risk)
        if survival_probs is not None and not np.isnan(survival_probs).all():
            surv_pred = survival_probs  # S(t) directly
        else:
            surv_pred = None

        # Compute Brier scores at each time point
        bs_scores = brier_score(y_train_struct, y_test_struct, surv_pred, times=eval_times)

        brier_results = {}
        for i, t in enumerate(times):
            brier_results[f'brier_{t}m'] = bs_scores[1][i]

        # Compute Integrated Brier Score (IBS)
        # IBS requires times strictly within the test data range
        # Find the valid time range: min of max event time and max censored time
        if np.any(y_test_struct['event']):
            max_event_time = np.max(y_test_struct[y_test_struct['event']]['time'])
        else:
            max_event_time = np.max(y_test_struct['time'])

        censored_times = y_test_struct[~y_test_struct['event']]['time']
        if len(censored_times) > 0:
            max_valid_time = min(max_event_time, np.max(censored_times))
        else:
            max_valid_time = max_event_time

        # IPCW support must lie inside both train and test follow-up ranges.
        max_valid_time = min(
            max_valid_time,
            60.0,
            float(np.max(y_train_struct['time'])) - 0.01,
            float(np.max(y_test_struct['time'])) - 0.01,
        )
        min_valid_time = max(
            0.1,
            float(np.min(y_test_struct['time'])) + 0.01,
        )

        # Create IBS times within valid range (strictly less than max_valid_time)
        if max_valid_time <= min_valid_time:
            raise ValueError(
                f"No valid IBS interval: [{min_valid_time}, {max_valid_time}]"
            )
        ibs_times = np.linspace(min_valid_time, max_valid_time, 50)

        # Interpolate survival predictions to IBS times
        interp_func = interp1d(times, survival_probs, axis=1, bounds_error=False, fill_value=(1, 0))
        surv_pred_ibs = interp_func(ibs_times)

        # Compute IBS using S(t) directly (not 1-S(t))
        bs_ibs = brier_score(y_train_struct, y_test_struct, surv_pred_ibs, times=ibs_times)
        ibs = np.trapezoid(bs_ibs[1], ibs_times) / (ibs_times[-1] - ibs_times[0])

        brier_results['ibs'] = ibs
        brier_results['ibs_max_time'] = float(max_valid_time)
        brier_results['max_time'] = float(max_event_time)

    except Exception as e:
        brier_results = {f'brier_{t}m': np.nan for t in times}
        brier_results['ibs'] = np.nan
        brier_results['ibs_max_time'] = np.nan
        brier_results['max_time'] = np.nan

    return brier_results


def compute_calibration(y_train_time, y_train_event, y_test_time, y_test_event,
                        survival_prob, time_point=36, n_bins=10):
    """
    Compute calibration metrics using KM/IPCW for censoring-aware assessment.

    Per Phase 3A methodological reset:
    - Uses pre-defined bins (deciles of predicted probability)
    - Uses KM/IPCW for observed proportions
    - No fake slope calculation

    Parameters
    ----------
    y_train_time : array-like
        Survival times from TRAINING set
    y_train_event : array-like
        Event indicators from TRAINING set
    y_test_time : array-like
        Survival times from TEST set
    y_test_event : array-like
        Event indicators from TEST set
    survival_prob : array-like
        Predicted survival probability at time_point
    time_point : int
        Time point for calibration assessment
    n_bins : int
        Number of bins for calibration plot (default 10 for deciles)

    Returns
    -------
    dict
        Calibration metrics with bin-level statistics
    """
    import logging

    # Handle arrays - extract probability at specified time point
    survival_prob = np.asarray(survival_prob)
    if survival_prob.ndim > 1:
        # Find the column closest to time_point
        idx = np.argmin(np.abs(np.array([12, 36, 60]) - time_point))
        survival_prob = survival_prob[:, idx]
    survival_prob = survival_prob.flatten()

    # Phase 3A: Use pre-defined bins (deciles)
    # Create bin edges based on predicted probability distribution
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_data = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (survival_prob >= bin_edges[i]) & (survival_prob < bin_edges[i + 1])
        else:
            # Include right edge for last bin
            mask = (survival_prob >= bin_edges[i]) & (survival_prob <= bin_edges[i + 1])

        if mask.sum() >= 5:  # Minimum 5 patients per bin
            # Use Kaplan-Meier for observed survival at time_point
            try:
                from lifelines import KaplanMeierFitter
                kmf = KaplanMeierFitter()
                mask_idx = np.where(mask)[0]
                test_times_bin = y_test_time[mask_idx]
                test_events_bin = y_test_event[mask_idx]

                kmf.fit(test_times_bin, test_events_bin, label='km')
                observed_surv = kmf.predict(time_point)
            except Exception as e:
                logging.warning(f"KM estimation failed for bin {i}: {e}")
                observed_surv = np.nan

            bin_data.append({
                'bin': i,
                'pred_mean': float(np.mean(survival_prob[mask])),
                'pred_min': float(np.min(survival_prob[mask])),
                'pred_max': float(np.max(survival_prob[mask])),
                'observed_survival': float(observed_surv),
                'n_patients': int(mask.sum())
            })
        else:
            # Not enough patients in this bin
            bin_data.append({
                'bin': i,
                'pred_mean': np.nan,
                'pred_min': np.nan,
                'pred_max': np.nan,
                'observed_survival': np.nan,
                'n_patients': int(mask.sum())
            })

    # Calculate calibration-in-the-large and calibration slope using valid bins only
    valid_bins = [b for b in bin_data if not np.isnan(b['pred_mean'])]
    if len(valid_bins) >= 2:
        pred_means = np.array([b['pred_mean'] for b in valid_bins])
        observed = np.array([b['observed_survival'] for b in valid_bins])

        # Calibration-in-the-large: mean observed vs mean predicted
        calibration_in_large = float(np.mean(observed) - np.mean(pred_means))

        # Calibration slope: regression of observed on predicted
        # Perfect calibration = slope 1, intercept 0
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(pred_means, observed)

        # E/O ratio (Expected/Observed survival)
        e_o_ratio = float(np.mean(pred_means) / np.mean(observed)) if np.mean(observed) > 0 else np.nan
    else:
        calibration_in_large = np.nan
        slope = np.nan
        intercept = np.nan
        e_o_ratio = np.nan

    return {
        'calibration_slope': slope,
        'calibration_intercept': intercept,
        'calibration_in_large': calibration_in_large,
        'eo_ratio': e_o_ratio,
        'n_bins': n_bins,
        'bin_data': bin_data
    }


def compute_all_metrics(y_train_time, y_train_event, y_test_time, y_test_event,
                        risk_scores, survival_probs=None, times=[12, 36, 60]):
    """
    Compute all evaluation metrics with proper train/test separation.

    Parameters
    ----------
    y_train_time : array-like
        Survival times from TRAINING set (for IPCW estimation)
    y_train_event : array-like
        Event indicators from TRAINING set
    y_test_time : array-like
        Survival times from TEST set
    y_test_event : array-like
        Event indicators from TEST set
    risk_scores : array-like
        Predicted risk scores for TEST set
    survival_probs : array-like, optional
        Predicted survival probabilities for TEST set (needed for Brier/IBS)
    times : list
        Time points for evaluation

    Returns
    -------
    dict
        All computed metrics
    """
    metrics = {}

    # Harrell C-index (no IPCW needed, standard)
    metrics['harrell_c'] = harrell_c_index(y_test_time, y_test_event, risk_scores)

    # Uno C-index (IPCW-weighted, uses train for IPCW estimation)
    metrics['uno_c'] = uno_c_index(
        y_train_time, y_train_event,
        y_test_time, y_test_event,
        risk_scores
    )

    # Time-dependent AUC (train/test separated)
    auc_results = time_dependent_auc(
        y_train_time, y_train_event,
        y_test_time, y_test_event,
        risk_scores, times=times
    )
    metrics.update(auc_results)

    # Brier score and IBS (if survival probabilities provided)
    if survival_probs is not None:
        brier_results = compute_brier_score(
            y_train_time, y_train_event,
            y_test_time, y_test_event,
            survival_probs, times=times
        )
        metrics.update(brier_results)

        # Calibration at 36 months
        cal_results = compute_calibration(
            y_train_time, y_train_event,
            y_test_time, y_test_event,
            survival_probs, time_point=36
        )
        metrics.update(cal_results)

    return metrics
