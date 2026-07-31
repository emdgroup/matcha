"""Shared fixtures for calibration tests."""

import os
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTING_DATA_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "testing_data.csv"
)


# ---------------------------------------------------------------------------
# Random generator
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Molecule fixtures (from testing_data.csv)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def testing_df() -> pd.DataFrame:
    """Load the shared testing CSV once per session."""
    return pd.read_csv(TESTING_DATA_CSV)


@pytest.fixture(scope="session")
def smiles_list(testing_df) -> list[str]:
    """First 100 SMILES strings from testing_data.csv."""
    return testing_df["SMILES"].tolist()[:100]


@pytest.fixture(scope="session")
def mol_list(smiles_list) -> list[Mol]:
    """RDKit Mol objects derived from *smiles_list*."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    assert all(m is not None for m in mols)
    return mols


@pytest.fixture(scope="session")
def small_mol_list(testing_df) -> list[Mol]:
    """Subset of 10 molecules for quick tests."""
    smiles = testing_df["SMILES"].tolist()[:10]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    assert all(m is not None for m in mols)
    return mols


# ---------------------------------------------------------------------------
# ICP regression fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_calibration_data(rng):
    """Synthetic regression calibration data (50 samples, 1 task)."""
    y_true = rng.normal(loc=5.0, scale=1.0, size=(50, 1))
    noise = rng.normal(loc=0.0, scale=0.3, size=(50, 1))
    y_pred = y_true + noise
    y_error = np.abs(noise) + 0.01  # ensure positive
    return y_true, y_pred, y_error


@pytest.fixture()
def regression_calibration_data_1d(rng):
    """Synthetic regression calibration data as 1-D arrays."""
    y_true = rng.normal(loc=5.0, scale=1.0, size=50)
    noise = rng.normal(loc=0.0, scale=0.3, size=50)
    y_pred = y_true + noise
    y_error = np.abs(noise) + 0.01
    return y_true, y_pred, y_error


@pytest.fixture()
def multitask_regression_calibration_data(rng):
    """Synthetic multitask regression data (50 samples, 3 tasks) with NaN."""
    y_true = rng.normal(loc=5.0, scale=1.0, size=(50, 3))
    noise = rng.normal(loc=0.0, scale=0.3, size=(50, 3))
    y_pred = y_true + noise
    y_error = np.abs(noise) + 0.01
    # Insert some NaN in y_true for task 1 and 2
    y_true[0, 1] = np.nan
    y_true[5, 2] = np.nan
    y_true[10, 1] = np.nan
    return y_true, y_pred, y_error


# ---------------------------------------------------------------------------
# ICP classification fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def classification_calibration_data(rng):
    """Synthetic binary classification calibration data (50 samples, 1 task)."""
    y_true = rng.choice([0.0, 1.0], size=(50, 1))
    # Predictions are probabilities for class 1
    y_pred = np.clip(y_true + rng.normal(0, 0.2, size=(50, 1)), 0.01, 0.99)
    return y_true, y_pred


@pytest.fixture()
def classification_calibration_data_1d(rng):
    """Synthetic binary classification data as 1-D arrays."""
    y_true = rng.choice([0.0, 1.0], size=50)
    y_pred = np.clip(y_true + rng.normal(0, 0.2, size=50), 0.01, 0.99)
    return y_true, y_pred


@pytest.fixture()
def multitask_classification_calibration_data(rng):
    """Synthetic multitask binary classification data (50 samples, 2 tasks) with NaN."""
    y_true = rng.choice([0.0, 1.0], size=(50, 2))
    y_pred = np.clip(y_true + rng.normal(0, 0.2, size=(50, 2)), 0.01, 0.99)
    y_true[2, 1] = np.nan
    y_true[7, 0] = np.nan
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Error model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def em_regression_data(rng, mol_list):
    """Error model regression data using real molecules (100 samples, 1 task)."""
    n = len(mol_list)
    y_true = rng.normal(loc=5.0, scale=2.0, size=(n, 1))
    noise = rng.normal(loc=0.0, scale=0.5, size=(n, 1))
    y_pred = y_true + noise
    y_error = np.abs(noise) + 0.01
    return mol_list, y_true, y_pred, y_error


@pytest.fixture()
def em_classification_data(rng, mol_list):
    """Error model classification data using real molecules (100 samples, 1 task)."""
    n = len(mol_list)
    y_true = rng.choice([0.0, 1.0], size=(n, 1))
    y_pred = np.clip(y_true + rng.normal(0, 0.3, size=(n, 1)), 0.01, 0.99)
    y_error = np.abs(rng.normal(0, 0.1, size=(n, 1))) + 0.01
    return mol_list, y_true, y_pred, y_error
