"""Pytest configuration for Phase 3A tests."""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pytest


@pytest.fixture
def inner_cv_splits():
    """Generate inner CV splits for hyperparameter tuning tests.

    Per Phase 3A reset: all tune() methods require inner_cv_splits.
    """
    from prognostic_engine.inner_splits import generate_inner_splits
    return generate_inner_splits(list(range(50)), repeat=1, outer_fold=1)


@pytest.fixture
def sample_survival_data():
    """Generate sample survival data for testing.

    Returns:
        dict with times, events, and risk_scores
    """
    np.random.seed(42)
    n = 100

    # Generate survival times with ~30% event rate
    times = np.random.exponential(scale=30, size=n)
    events = np.random.binomial(1, 0.3, size=n)

    # Generate risk scores correlated with survival
    base_risk = -0.05 * times + np.random.normal(0, 0.5, n)
    risk_scores = base_risk - np.min(base_risk)  # Shift to positive

    return {
        'times': times,
        'events': events,
        'risk_scores': risk_scores,
        'n': n
    }


@pytest.fixture
def train_test_split(sample_survival_data):
    """Split survival data into train/test.

    Returns:
        dict with train and test subsets
    """
    np.random.seed(42)
    n = sample_survival_data['n']
    idx = np.random.permutation(n)
    split_point = int(0.7 * n)

    train_idx = idx[:split_point]
    test_idx = idx[split_point:]

    return {
        'train_times': sample_survival_data['times'][train_idx],
        'train_events': sample_survival_data['events'][train_idx],
        'test_times': sample_survival_data['times'][test_idx],
        'test_events': sample_survival_data['events'][test_idx],
        'test_risk_scores': sample_survival_data['risk_scores'][test_idx]
    }


@pytest.fixture
def sample_features():
    """Generate sample feature matrix for testing models.

    Returns:
        ndarray of shape (n_samples, n_features)
    """
    np.random.seed(42)
    n = 100
    n_features = 20

    # Generate correlated features
    X = np.random.randn(n, n_features)
    # Add some correlation structure
    X[:, 0] = 0.5 * X[:, 0] + 0.5 * np.random.randn(n)  # Age-like
    X[:, 1] = np.random.randint(0, 4, size=n)  # Stage-like
    X[:, 2:7] = np.random.exponential(scale=1, size=(n, 5))  # Gene-like

    return X
