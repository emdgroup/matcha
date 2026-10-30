"""Shared pytest fixtures for sklearn manager tests.

Provides molecule lists and label arrays used across all test modules
in this folder.  Every fixture that needs ``testing_data.csv`` goes here
so individual test files stay focused on the manager under test.
"""

import os

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

# =========================================================================
# Paths
# =========================================================================

TESTING_DATA_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "testing_data.csv"
)

# =========================================================================
# Raw-data fixtures
# =========================================================================


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
    """Single-task regression labels (N, 1) for the first 30 compounds."""
    return testing_df["Regression"].values[:30].reshape(-1, 1).astype(np.float64)


@pytest.fixture(scope="session")
def classification_y(testing_df) -> np.ndarray:
    """Single-task binary classification labels (N, 1) for the first 30 compounds."""
    return testing_df["Classification"].values[:30].reshape(-1, 1).astype(np.float64)


# =========================================================================
# Shared minimal training kwargs
# =========================================================================

BASE_TRAIN = dict(
    num_epochs=1,
    batch_size=32,
    accelerator="cpu",
    devices=1,
    early_stopping=False,
    stochastic_weight_averaging=False,
)

GRAPH_PE_OFF = dict(
    rwse_k=0,
    laplacian_k=0,
    elstatic_k=0,
    distmat_k=0,
    rrwp_k=0,
    num_virtual_nodes=0,
)
