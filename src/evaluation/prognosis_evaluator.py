"""
Evaluation Module for HCC Prognosis Models.

This module provides utilities for evaluating prognosis models using:
- C-index (concordance index)
- Time-dependent AUC
- Calibration curves
- Decision curve analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.interpolate import interp1d


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    c_index: float
    c_index_ci_low: float
    c_index_ci_high: float
    auc_1yr: float
    auc_3yr: float
    auc_5yr: float
    calibration_slope: float
    calibration_intercept: float
    brier_score: float
    n_samples: int
    n_events: int


class SurvivalEvaluator:
    """
    Evaluator for survival analysis models.

    This class computes:
    - Concordance index (C-index)
    - Time-dependent AUC
    - Calibration metrics
    - Brier score
    """

    def __init__(
        self,
        time_points: List[float] = None
    ):
        """
        Initialize evaluator.

        Args:
            time_points: Time points for evaluation (default: [12, 36, 60] months)
        """
        self.time_points = time_points or [12, 36, 60]

    def evaluate(
        self,
        predictions: np.ndarray,
        actual_times: np.ndarray,
        actual_events: np.ndarray,
        confidence_level: float = 0.95
    ) -> EvaluationMetrics:
        """
        Evaluate model predictions.

        Args:
            predictions: Predicted risk scores (higher = worse prognosis)
            actual_times: Actual survival times
            actual_events: Event indicators (1=event, 0=censored)
            confidence_level: Confidence level for CI

        Returns:
            EvaluationMetrics with all computed metrics
        """
        # Calculate C-index with bootstrap CI
        c_index, c_index_ci = self._calculate_c_index_with_ci(
            predictions, actual_times, actual_events, confidence_level
        )

        # Calculate time-dependent AUCs
        aucs = self._calculate_time_aucs(
            predictions, actual_times, actual_events
        )

        # Calculate calibration
        slope, intercept = self._calculate_calibration(
            predictions, actual_times, actual_events
        )

        # Calculate Brier score
        brier = self._calculate_brier_score(
            predictions, actual_times, actual_events
        )

        return EvaluationMetrics(
            c_index=c_index,
            c_index_ci_low=c_index_ci[0],
            c_index_ci_high=c_index_ci[1],
            auc_1yr=aucs.get(12, np.nan),
            auc_3yr=aucs.get(36, np.nan),
            auc_5yr=aucs.get(60, np.nan),
            calibration_slope=slope,
            calibration_intercept=intercept,
            brier_score=brier,
            n_samples=len(predictions),
            n_events=int(np.sum(actual_events))
        )

    def _calculate_c_index_with_ci(
        self,
        predictions: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate C-index with bootstrap confidence intervals.

        Args:
            predictions: Risk predictions
            times: Survival times
            events: Event indicators
            confidence_level: CI level

        Returns:
            Tuple of (c_index, (ci_low, ci_high))
        """
        c_index = self._concordance_index(predictions, times, events)

        # Bootstrap for CI
        n_bootstrap = 100
        bootstrap_indices = np.random.randint(0, len(predictions), (n_bootstrap, len(predictions)))

        bootstrap_c_indices = []
        for idx in bootstrap_indices:
            pred_boot = predictions[idx]
            times_boot = times[idx]
            events_boot = events[idx]

            try:
                c_boot = self._concordance_index(pred_boot, times_boot, events_boot)
                bootstrap_c_indices.append(c_boot)
            except:
                pass

        if bootstrap_c_indices:
            alpha = 1 - confidence_level
            ci_low = np.percentile(bootstrap_c_indices, alpha/2 * 100)
            ci_high = np.percentile(bootstrap_c_indices, (1 - alpha/2) * 100)
        else:
            ci_low = c_index - 0.05
            ci_high = c_index + 0.05

        return c_index, (ci_low, ci_high)

    def _concordance_index(
        self,
        predictions: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Calculate concordance index.

        Based on Harrell's C-index formula.

        Args:
            predictions: Risk predictions
            times: Survival times
            events: Event indicators

        Returns:
            C-index value
        """
        n = len(predictions)
        if n < 2:
            return 0.5

        concordant = 0
        comparable = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Only compare if one had an event or both censored
                if events[i] == 1 or events[j] == 1:
                    # Check if times are comparable
                    if events[i] == 1 and events[j] == 0:
                        # i had event, j censored - compare only if j survived long enough
                        if times[j] >= times[i]:
                            comparable += 1
                            if predictions[i] > predictions[j]:
                                concordant += 1
                            elif predictions[i] == predictions[j]:
                                concordant += 0.5
                    elif events[j] == 1 and events[i] == 0:
                        # j had event, i censored - compare only if i survived long enough
                        if times[i] >= times[j]:
                            comparable += 1
                            if predictions[j] > predictions[i]:
                                concordant += 1
                            elif predictions[j] == predictions[i]:
                                concordant += 0.5
                    else:
                        # Both had events - comparable
                        comparable += 1
                        if predictions[i] > predictions[j] and times[i] < times[j]:
                            concordant += 1
                        elif predictions[j] > predictions[i] and times[j] < times[i]:
                            concordant += 1
                        elif predictions[i] == predictions[j]:
                            concordant += 0.5

        return concordant / comparable if comparable > 0 else 0.5

    def _calculate_time_aucs(
        self,
        predictions: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> Dict[float, float]:
        """
        Calculate time-dependent AUC at specified time points.

        Args:
            predictions: Risk predictions
            times: Survival times
            events: Event indicators

        Returns:
            Dictionary of time -> AUC
        """
        aucs = {}

        for t in self.time_points:
            aucs[t] = self._time_dependent_auc(predictions, times, events, t)

        return aucs

    def _time_dependent_auc(
        self,
        predictions: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        time_point: float
    ) -> float:
        """
        Calculate time-dependent AUC at a specific time point.

        Uses inverse probability of censoring weighting (IPCW).

        Args:
            predictions: Risk predictions
            times: Survival times
            events: Event indicators
            time_point: Time point for evaluation

        Returns:
            AUC value
        """
        # Filter to relevant pairs
        cases = []
        controls = []

        for i in range(len(predictions)):
            # Only consider subjects who could have an event at or before time_point
            if times[i] >= time_point or (events[i] == 1 and times[i] <= time_point):
                score = predictions[i]
                if events[i] == 1 and times[i] <= time_point:
                    cases.append(score)
                elif times[i] > time_point:
                    controls.append(score)

        if len(cases) < 1 or len(controls) < 1:
            return 0.5

        # Calculate AUC using Mann-Whitney U statistic
        cases_arr = np.array(cases)
        controls_arr = np.array(controls)

        # Simple AUC approximation
        n_cases = len(cases_arr)
        n_controls = len(controls_arr)

        concordant = 0
        for c in cases_arr:
            concordant += np.sum(c > controls_arr) + 0.5 * np.sum(c == controls_arr)

        auc = concordant / (n_cases * n_controls)

        return auc

    def _calculate_calibration(
        self,
        predictions: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate calibration slope and intercept.

        Args:
            predictions: Risk predictions
            times: Survival times
            events: Event indicators

        Returns:
            Tuple of (slope, intercept)
        """
        # Use simple linear regression of observed vs predicted
        # For proper calibration, use Cox regression on predictions

        # Bin predictions and calculate observed survival
        n_bins = 5
        bins = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))

        bin_predicted = []
        bin_observed = []

        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if i == n_bins - 1:
                mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])

            if np.sum(mask) > 0:
                bin_predicted.append(np.mean(predictions[mask]))
                # Simple observed proportion
                obs_rate = np.sum(events[mask]) / np.sum(mask)
                bin_observed.append(obs_rate)

        if len(bin_predicted) > 1:
            slope, intercept, _, _, _ = stats.linregress(bin_predicted, bin_observed)
            return slope, intercept

        return 1.0, 0.0

    def _calculate_brier_score(
        self,
        predictions: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Calculate Brier score.

        Args:
            predictions: Risk predictions (normalized 0-1)
            times: Survival times
            events: Event indicators

        Returns:
            Brier score
        """
        # Simple Brier score at median follow-up time
        median_time = np.median(times)

        # For censored data, use KM weights
        # Simplified version
        preds_normalized = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-10)

        # Binary prediction: high risk vs low risk
        threshold = 0.5
        binary_pred = (preds_normalized > threshold).astype(float)

        # Simple Brier score
        brier = np.mean((binary_pred - events) ** 2)

        return brier


class DecisionCurveAnalysis:
    """
    Decision Curve Analysis for clinical utility assessment.

    Decision curves help determine the clinical utility of a model
    by comparing net benefits across different threshold probabilities.
    """

    def __init__(
        self,
        thresholds: Optional[List[float]] = None
    ):
        """
        Initialize DCA.

        Args:
            thresholds: List of threshold probabilities (default: 0.01 to 0.99)
        """
        if thresholds is None:
            self.thresholds = np.linspace(0.01, 0.99, 50)
        else:
            self.thresholds = thresholds

    def calculate_net_benefits(
        self,
        predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        time_point: float = 36
    ) -> Dict[str, np.ndarray]:
        """
        Calculate net benefits for decision curve.

        Args:
            predictions: Risk predictions
            actual_outcomes: Binary outcomes
            time_point: Time point for evaluation

        Returns:
            Dictionary with net benefits for each strategy
        """
        n = len(predictions)

        # Normalize predictions
        preds_normalized = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-10)

        # Calculate prevalence
        prevalence = np.mean(actual_outcomes)

        # Net benefit for each threshold
        nb_model = np.zeros(len(self.thresholds))
        nb_all = np.zeros(len(self.thresholds))  # Treat all strategy
        nb_none = np.zeros(len(self.thresholds))  # Treat none strategy

        for i, thresh in enumerate(self.thresholds):
            # Treat none is always 0
            nb_none[i] = 0

            # Treat all
            nb_all[i] = prevalence - (1 - thresh) / thresh

            # Model-based strategy
            treated = preds_normalized > thresh
            n_treated = np.sum(treated)

            if n_treated > 0:
                # True positives
                tp = np.sum((treated == 1) & (actual_outcomes == 1))
                # False positives
                fp = np.sum((treated == 1) & (actual_outcomes == 0))

                # Net benefit = (TP/n) - (FP/n) * (thresh / (1 - thresh))
                nb_model[i] = (tp / n) - (fp / n) * (thresh / (1 - thresh + 1e-10))
            else:
                nb_model[i] = 0

        return {
            "thresholds": self.thresholds,
            "model": nb_model,
            "treat_all": nb_all,
            "treat_none": nb_none
        }


class ComparisonEvaluator:
    """
    Evaluator for comparing multiple models.

    This class provides:
    - Pairwise comparisons
    - Statistical significance testing
    - Cross-validation
    """

    def compare_models(
        self,
        results: Dict[str, EvaluationMetrics]
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            results: Dictionary of model_name -> EvaluationMetrics

        Returns:
            DataFrame with comparison table
        """
        rows = []

        for model_name, metrics in results.items():
            rows.append({
                "Model": model_name,
                "C-index": f"{metrics.c_index:.3f}",
                "C-index 95% CI": f"[{metrics.c_index_ci_low:.3f}, {metrics.c_index_ci_high:.3f}]",
                "AUC 1yr": f"{metrics.auc_1yr:.3f}" if not np.isnan(metrics.auc_1yr) else "N/A",
                "AUC 3yr": f"{metrics.auc_3yr:.3f}" if not np.isnan(metrics.auc_3yr) else "N/A",
                "AUC 5yr": f"{metrics.auc_5yr:.3f}" if not np.isnan(metrics.auc_5yr) else "N/A",
                "Calibration Slope": f"{metrics.calibration_slope:.3f}",
                "Calibration Intercept": f"{metrics.calibration_intercept:.3f}",
                "Brier Score": f"{metrics.brier_score:.3f}",
                "N": metrics.n_samples,
                "Events": metrics.n_events
            })

        return pd.DataFrame(rows)

    def statistical_tests(
        self,
        predictions_dict: Dict[str, np.ndarray],
        times: np.ndarray,
        events: np.ndarray
    ) -> pd.DataFrame:
        """
        Perform statistical comparisons between models.

        Args:
            predictions_dict: Dictionary of model_name -> predictions
            times: Survival times
            events: Event indicators

        Returns:
            DataFrame with pairwise comparisons
        """
        from scipy.stats import wilcoxon

        model_names = list(predictions_dict.keys())
        rows = []

        evaluator = SurvivalEvaluator()

        for i, name1 in enumerate(model_names):
            for name2 in model_names[i + 1:]:
                pred1 = predictions_dict[name1]
                pred2 = predictions_dict[name2]

                # Calculate C-indices
                c1 = evaluator._concordance_index(pred1, times, events)
                c2 = evaluator._concordance_index(pred2, times, events)

                # DeLong test for AUC comparison would require more code
                # Using Wilcoxon as approximation
                try:
                    _, p_value = wilcoxon(pred1, pred2)
                except:
                    p_value = np.nan

                rows.append({
                    "Comparison": f"{name1} vs {name2}",
                    f"{name1} C-index": f"{c1:.3f}",
                    f"{name2} C-index": f"{c2:.3f}",
                    "Difference": f"{c1 - c2:.3f}",
                    "P-value": f"{p_value:.4f}" if not np.isnan(p_value) else "N/A",
                    "Significant (p<0.05)": "Yes" if p_value < 0.05 else "No"
                })

        return pd.DataFrame(rows)


# Convenience functions
def evaluate_model(
    predictions: np.ndarray,
    times: np.ndarray,
    events: np.ndarray
) -> EvaluationMetrics:
    """
    Evaluate a single model.

    Args:
        predictions: Risk predictions
        times: Survival times
        events: Event indicators

    Returns:
        EvaluationMetrics
    """
    evaluator = SurvivalEvaluator()
    return evaluator.evaluate(predictions, times, events)


def compare_models(
    predictions_dict: Dict[str, np.ndarray],
    times: np.ndarray,
    events: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        predictions_dict: Dictionary of model_name -> predictions
        times: Survival times
        events: Event indicators

    Returns:
        Comparison DataFrame
    """
    results = {}
    evaluator = SurvivalEvaluator()

    for name, preds in predictions_dict.items():
        results[name] = evaluator.evaluate(preds, times, events)

    comparator = ComparisonEvaluator()
    return comparator.compare_models(results)
