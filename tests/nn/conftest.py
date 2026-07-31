"""Shared fixtures for matcha.nn tests."""

import os

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTING_DATA_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "testing_data.csv"
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_everything():
    """Set random seeds for reproducibility in every test."""
    torch.manual_seed(42)
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Force CPU
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> torch.device:
    """Always use CPU to avoid GPU-related issues in CI."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Generic tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def batch_size() -> int:
    return 16


@pytest.fixture()
def input_dim() -> int:
    return 32


@pytest.fixture()
def output_dim() -> int:
    return 16


@pytest.fixture()
def random_input(batch_size, input_dim) -> torch.Tensor:
    """Random float tensor of shape (batch_size, input_dim)."""
    return torch.randn(batch_size, input_dim)


@pytest.fixture()
def binary_targets(batch_size) -> torch.Tensor:
    """Binary target labels (0 or 1) of shape (batch_size, 1)."""
    return torch.randint(0, 2, (batch_size, 1)).float()


@pytest.fixture()
def regression_targets(batch_size) -> torch.Tensor:
    """Regression target values of shape (batch_size, 1)."""
    return torch.randn(batch_size, 1)


@pytest.fixture()
def multitask_targets(batch_size) -> torch.Tensor:
    """Multitask regression targets with some NaN values, shape (batch_size, 3)."""
    targets = torch.randn(batch_size, 3)
    # sprinkle some NaNs
    targets[0, 1] = float("nan")
    targets[3, 0] = float("nan")
    targets[7, 2] = float("nan")
    return targets


# ---------------------------------------------------------------------------
# Testing CSV data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def testing_df() -> pd.DataFrame:
    """Load the shared testing CSV once per session."""
    return pd.read_csv(TESTING_DATA_CSV)
