"""Shared fixtures for datamodule tests."""

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
# Raw data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def testing_df() -> pd.DataFrame:
    """Load the shared testing CSV once per session."""
    return pd.read_csv(TESTING_DATA_CSV)


@pytest.fixture(scope="session")
def smiles_list(testing_df) -> list[str]:
    """First 30 SMILES strings from testing_data.csv."""
    return testing_df["SMILES"].tolist()[:30]


@pytest.fixture(scope="session")
def mol_list(smiles_list) -> list[Mol]:
    """RDKit Mol objects derived from *smiles_list*."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    assert all(m is not None for m in mols)
    return mols


@pytest.fixture(scope="session")
def regression_y(testing_df) -> np.ndarray:
    """Regression labels (N, 1) for the first 30 compounds."""
    return testing_df["Regression"].values[:30].reshape(-1, 1).astype(np.float64)


@pytest.fixture(scope="session")
def classification_y(testing_df) -> np.ndarray:
    """Binary classification labels (N, 1) for the first 30 compounds."""
    return testing_df["Classification"].values[:30].reshape(-1, 1).astype(np.float64)


@pytest.fixture(scope="session")
def small_mol_list(mol_list) -> list[Mol]:
    """Subset of 5 molecules for quick tests."""
    return mol_list[:5]


@pytest.fixture(scope="session")
def small_regression_y(regression_y) -> np.ndarray:
    """Regression labels matching *small_mol_list*."""
    return regression_y[:5]


@pytest.fixture(scope="session")
def small_classification_y(classification_y) -> np.ndarray:
    """Classification labels matching *small_mol_list*."""
    return classification_y[:5]


@pytest.fixture(scope="session")
def bound_mask_exact() -> list[str]:
    """Bound mask with all exact values – 30 entries."""
    return ["="] * 30


@pytest.fixture(scope="session")
def bound_mask_mixed() -> list[str]:
    """Bound mask with mixed values – 30 entries."""
    masks = []
    for i in range(30):
        if i % 5 == 0:
            masks.append("<")
        elif i % 7 == 0:
            masks.append(">")
        else:
            masks.append("=")
    return masks
