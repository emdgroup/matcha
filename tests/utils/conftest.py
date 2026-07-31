"""Shared pytest fixtures for utility tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def regression_labels(rng) -> np.ndarray:
    """Continuous regression labels (100,)."""
    return rng.normal(5.0, 1.0, size=100)


@pytest.fixture()
def regression_predictions(regression_labels, rng) -> np.ndarray:
    """Noisy regression predictions correlated with *regression_labels*."""
    return regression_labels + rng.normal(0, 0.3, size=regression_labels.shape)


@pytest.fixture()
def classification_labels(rng) -> np.ndarray:
    """Binary labels (100,) with roughly balanced classes."""
    return rng.choice([0.0, 1.0], size=100, p=[0.5, 0.5])


@pytest.fixture()
def classification_predictions(classification_labels) -> np.ndarray:
    """Perfect hard predictions matching *classification_labels*."""
    return classification_labels.copy()


@pytest.fixture()
def classification_probabilities(classification_labels, rng) -> np.ndarray:
    """Probabilities correlated with the true class."""
    probs = np.where(
        classification_labels == 1,
        rng.uniform(0.6, 1.0, size=classification_labels.shape),
        rng.uniform(0.0, 0.4, size=classification_labels.shape),
    )
    return probs


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Small DataFrame with a date column and numeric column for splitting tests."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": range(100),
            "label": np.random.default_rng(0).normal(size=100),
        }
    )
