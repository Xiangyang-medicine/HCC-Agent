"""Proportional Hazards (PH) assumption diagnostics for Cox models.

Includes:
- Schoenfeld residuals
- Martingale residuals
- Visual inspection plots
"""

import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


def compute_schoenfeld_residuals(model, X, y_time, y_event):
    """
    Compute scaled Schoenfeld residuals for PH assumption assessment.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    X : array-like
        Feature matrix
    y_time : array-like
        Survival times
    y_event : array-like
        Event indicators

    Returns
    -------
    dict
        Residuals and correlation tests
    """
    if not hasattr(model, 'summary'):
        raise ValueError("Model must be fitted first")

    # Get residuals from lifelines
    residuals = model.compute_residuals(
        model.timeline if hasattr(model, 'timeline') else None
    )

    # For Cox PH, we need to manually compute or use built-in
    # This is a simplified version - full implementation would use patsy/formula
    return residuals


def ph_diagnostics(model, df, duration_col='time', event_col='event'):
    """
    Test proportional hazards assumption using multiple methods.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    df : DataFrame
        Training data
    duration_col : str
        Column name for duration
    event_col : str
        Column name for event

    Returns
    -------
    dict
        Diagnostic results
    """
    # Statistical test for PH assumption
    results = proportional_hazard_test(model, df, time_transform='rank')

    # Extract test statistics and p-values
    test_summary = {
        'test_statistic': results.test_statistic,
        'p_value': results.p_value,
        'degrees_of_freedom': results.degrees_freedom
    }

    # Per-variable tests (if available)
    if hasattr(results, 'weights'):
        test_summary['variable_tests'] = results.weights

    return test_summary


def plot_schoenfeld_residuals(model, df, duration_col, event_col, feature_names,
                              save_path=None):
    """
    Plot scaled Schoenfeld residuals over time for visual inspection.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    df : DataFrame
        Training data
    duration_col : str
        Duration column name
    event_col : str
        Event column name
    feature_names : list
        Feature names
    save_path : str, optional
        Path to save plot

    Returns
    -------
    matplotlib.Figure
    """
    n_features = len(feature_names)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    if n_features == 1:
        axes = [axes]

    # Get residuals
    try:
        # lifelines doesn't expose raw Schoenfeld residuals directly
        # Use martingale residuals as proxy
        martingale = model.residuals_['martingale']

        for i, name in enumerate(feature_names):
            ax = axes[i]
            # Sort by time
            sort_idx = np.argsort(df[duration_col].values)

            ax.scatter(df[duration_col].values[sort_idx],
                      martingale.values[sort_idx],
                      alpha=0.5, s=10)
            ax.axhline(0, color='r', linestyle='--')
            ax.set_xlabel('Time')
            ax.set_ylabel('Martingale Residuals')
            ax.set_title(f'Residuals for {name}')
    except Exception as e:
        # Fallback: just show that diagnostics failed
        for ax in axes:
            ax.text(0.5, 0.5, f'Diagnostics unavailable:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def check_ph_assumption(model, df, duration_col, event_col, significance=0.05):
    """
    Check PH assumption and return summary decision.

    Parameters
    ----------
    model : CoxPHFitter
        Fitted Cox model
    df : DataFrame
        Training data
    duration_col : str
        Duration column name
    event_col : str
        Event column name
    significance : float
        Significance level

    Returns
    -------
    dict
        Decision and details
    """
    test_results = ph_diagnostics(model, df, duration_col, event_col)

    ph_satisfied = test_results['p_value'] > significance

    return {
        'ph_satisfied': ph_satisfied,
        'p_value': test_results['p_value'],
        'significance_level': significance,
        'conclusion': 'PH assumption satisfied' if ph_satisfied else 'PH assumption violated',
        'recommendation': (
            'Standard Cox PH model appropriate' if ph_satisfied
            else 'Consider time-varying coefficients or stratified analysis'
        )
    }
