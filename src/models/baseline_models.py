"""
Baseline Prognosis Models for HCC.

This module implements traditional and deep learning baseline models
for survival prediction to compare against the LLM agent system.

Models:
- Cox Proportional Hazards (Cox PH)
- DeepSurv (deep learning survival model)
- Random survival forest
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import warnings


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    risk_scores: np.ndarray
    survival_probabilities: Optional[np.ndarray] = None
    risk_groups: Optional[np.ndarray] = None


class CoxProportionalHazards:
    """
    Cox Proportional Hazards Model.

    This is a simplified implementation of Cox PH for survival prediction.
    For production, consider using lifelines or pycox libraries.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize Cox PH model.

        Args:
            alpha: Regularization parameter
        """
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.baseline_hazard = None
        self.coef_ = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray
    ) -> "CoxProportionalHazards":
        """
        Fit the Cox model.

        Args:
            X: Feature matrix
            time: Survival times
            event: Event indicators

        Returns:
            self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit using gradient descent (simplified)
        n_features = X_scaled.shape[1]

        # Initialize coefficients
        self.coef_ = np.zeros(n_features)

        # Gradient descent optimization
        learning_rate = 0.01
        max_iterations = 500

        for iteration in range(max_iterations):
            # Calculate linear predictor
            eta = X_scaled @ self.coef_

            # Gradient (simplified partial likelihood)
            gradient = np.zeros(n_features)

            for i in range(len(X)):
                if event[i] == 1:
                    # Event occurred for this individual
                    risk_set = np.where(time >= time[i])[0]
                    if len(risk_set) > 0:
                        risk_scores = eta[risk_set]
                        max_risk = np.max(risk_scores)
                        exp_risk = np.exp(risk_scores - max_risk)
                        exp_sum = np.sum(exp_risk)

                        if exp_sum > 0:
                            for j in risk_set:
                                weight = np.exp(eta[j] - max_risk) / exp_sum
                                gradient += X_scaled[i] - X_scaled[j] * weight

            # Update coefficients
            self.coef_ += learning_rate * gradient / len(X)

            # L2 regularization
            self.coef_ *= (1 - self.alpha * learning_rate)

        # Estimate baseline hazard
        self._estimate_baseline_hazard(X_scaled, time, event)

        self.is_fitted = True
        return self

    def _estimate_baseline_hazard(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray
    ):
        """Estimate baseline cumulative hazard."""
        unique_times = np.unique(time[event == 1])
        baseline_hazard = np.zeros(len(unique_times))

        for i, t in enumerate(unique_times):
            at_risk = np.where(time >= t)[0]
            n_at_risk = len(at_risk)
            n_events = np.sum((time == t) & (event == 1))

            if n_at_risk > 0:
                baseline_hazard[i] = n_events / n_at_risk

        self.baseline_hazard = (unique_times, np.cumsum(baseline_hazard))

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores.

        Args:
            X: Feature matrix

        Returns:
            Risk scores (higher = worse)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.coef_

    def predict_survival(
        self,
        X: np.ndarray,
        time_points: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict survival probability.

        Args:
            X: Feature matrix
            time_points: Time points for prediction

        Returns:
            Tuple of (time_points, survival_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        risk_scores = self.predict_risk(X)
        baseline_times, baseline_hazard = self.baseline_hazard

        if time_points is None:
            time_points = baseline_times

        survival_probs = np.zeros((len(X), len(time_points)))

        for i in range(len(X)):
            cumulative_hazard = np.interp(
                time_points, baseline_times,
                baseline_hazard * np.exp(risk_scores[i])
            )
            survival_probs[i] = np.exp(-cumulative_hazard)

        return time_points, survival_probs


class DeepSurvModel:
    """
    Simplified DeepSurv Implementation.

    This is a simplified version of DeepSurv for survival analysis.
    For production, consider using pycox library.

    Uses a neural network with negative partial likelihood loss.
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [32, 16],
        learning_rate: float = 0.001,
        max_epochs: int = 100
    ):
        """
        Initialize DeepSurv model.

        Args:
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            max_epochs: Maximum training epochs
        """
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.scaler = StandardScaler()
        self.weights = None
        self.biases = None
        self.is_fitted = False

    def _initialize_network(self, n_features: int):
        """Initialize network weights."""
        layer_sizes = [n_features] + self.hidden_sizes + [1]

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(float)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:  # Not output layer
                h = self._relu(h)
        return h

    def _negative_partial_log_likelihood(
        self,
        predictions: np.ndarray,
        time: np.ndarray,
        event: np.ndarray
    ) -> float:
        """Calculate negative partial log likelihood."""
        risk_scores = predictions.flatten()

        # Sort by time (descending for risk set calculation)
        order = np.argsort(-time)
        risk_scores_sorted = risk_scores[order]
        event_sorted = event[order]
        time_sorted = time[order]

        nll = 0.0
        for i in range(len(risk_scores)):
            if event_sorted[i] == 1:
                # Event occurred
                risk_set = np.where(time_sorted >= time_sorted[i])[0]
                log_risk_sum = np.log(
                    np.sum(np.exp(risk_scores_sorted[risk_set] - np.max(risk_scores_sorted[risk_set])))
                    + 1e-10
                ) + np.max(risk_scores_sorted[risk_set])
                nll -= risk_scores_sorted[i] - log_risk_sum

        return nll / np.sum(event)

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray
    ) -> "DeepSurvModel":
        """
        Fit the DeepSurv model.

        Args:
            X: Feature matrix
            time: Survival times
            event: Event indicators

        Returns:
            self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize network
        self._initialize_network(X_scaled.shape[1])

        # Training with gradient descent
        for epoch in range(self.max_epochs):
            # Forward pass
            predictions = self._forward(X_scaled)

            # Calculate loss (simplified)
            loss = self._negative_partial_log_likelihood(predictions, time, event)

            # Numerical gradient approximation for weights
            eps = 1e-5
            for l in range(len(self.weights)):
                for w_i in range(self.weights[l].shape[0]):
                    for w_j in range(self.weights[l].shape[1]):
                        # Gradient approximation
                        self.weights[l][w_i, w_j] += eps
                        pred_plus = self._forward(X_scaled)
                        loss_plus = self._negative_partial_log_likelihood(pred_plus, time, event)

                        self.weights[l][w_i, w_j] -= 2 * eps
                        pred_minus = self._forward(X_scaled)
                        loss_minus = self._negative_partial_log_likelihood(pred_minus, time, event)

                        gradient = (loss_plus - loss_minus) / (2 * eps)
                        self.weights[l][w_i, w_j] += eps  # Restore

                        # Update
                        self.weights[l][w_i, w_j] -= self.learning_rate * gradient

        self.is_fitted = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler.transform(X)
        predictions = self._forward(X_scaled)
        return predictions.flatten()


class SimpleSurvivalPredictor:
    """
    Simple baseline using logistic regression for binary survival outcome.

    This is a simple but robust baseline that:
    - Binarizes survival (e.g., alive at 1 year vs dead)
    - Uses logistic regression
    - Works well for comparison
    """

    def __init__(self, threshold_months: float = 12):
        """
        Initialize predictor.

        Args:
            threshold_months: Time threshold for binary outcome
        """
        self.threshold_months = threshold_months
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray
    ) -> "SimpleSurvivalPredictor":
        """
        Fit the model.

        Args:
            X: Feature matrix
            time: Survival times
            event: Event indicators

        Returns:
            self
        """
        X_scaled = self.scaler.fit_transform(X)

        # Create binary outcome: 1 if dead before threshold, 0 otherwise
        # For censored patients who survived past threshold, treat as 0
        # For censored patients before threshold, exclude
        mask = (event == 1) | (time >= self.threshold_months)
        y = (time < self.threshold_months) & (event == 1)
        y = y.astype(int)

        X_train = X_scaled[mask]
        y_train = y[mask]

        self.model.fit(X_train, y_train)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk (probability of event)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


def prepare_features(
    df: pd.DataFrame,
    gene_columns: List[str],
    categorical_cols: List[str] = ["stage", "grade", "gender"]
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare features from TCGA DataFrame.

    Args:
        df: TCGA DataFrame
        gene_columns: List of gene expression columns
        categorical_cols: Categorical columns to encode

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    features = []
    feature_names = []

    # Process categorical columns
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].fillna("Unknown"))
            features.append(encoded.reshape(-1, 1))
            feature_names.append(col)

    # Process numerical columns
    numerical_cols = ["age", "afp_level", "albumin", "bilirubin"]
    for col in numerical_cols:
        if col in df.columns:
            features.append(df[col].fillna(df[col].median()).values.reshape(-1, 1))
            feature_names.append(col)

    # Process gene expression
    for col in gene_columns:
        if col in df.columns:
            features.append(df[col].fillna(0).values.reshape(-1, 1))
            feature_names.append(col)

    # Combine all features
    X = np.hstack(features)

    return X, feature_names


class ModelFactory:
    """Factory for creating baseline models."""

    @staticmethod
    def create(model_type: str, **kwargs) -> Any:
        """
        Create a baseline model.

        Args:
            model_type: Type of model ('cox', 'deepsurv', 'simple')
            **kwargs: Model-specific arguments

        Returns:
            Model instance
        """
        if model_type == "cox":
            return CoxProportionalHazards(**kwargs)
        elif model_type == "deepsurv":
            return DeepSurvModel(**kwargs)
        elif model_type == "simple":
            return SimpleSurvivalPredictor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    time = np.random.exponential(30, n_samples)
    event = np.random.binomial(1, 0.5, n_samples)

    # Test Cox model
    cox = CoxProportionalHazards()
    cox.fit(X, time, event)
    risk_scores = cox.predict_risk(X)
    print(f"Cox C-index (approx): {np.corrcoef(risk_scores, -time)[0, 1]:.3f}")

    # Test Simple model
    simple = SimpleSurvivalPredictor(threshold_months=12)
    simple.fit(X, time, event)
    risk_simple = simple.predict_risk(X)
    print(f"Simple model fitted successfully")
