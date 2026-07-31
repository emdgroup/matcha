"""Shared fixtures for explainability tests."""

import os
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.explainability.lime import LIME

from matcha.explainability.explainer import MatchaExplainer


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
# Testing data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def testing_df() -> pd.DataFrame:
    """Load the shared testing CSV once per session."""
    return pd.read_csv(TESTING_DATA_CSV)


# ---------------------------------------------------------------------------
# Molecule fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def smiles_list(testing_df) -> list[str]:
    """First 50 SMILES strings from testing_data.csv."""
    return testing_df["SMILES"].tolist()[:50]


@pytest.fixture(scope="session")
def mol_list(smiles_list) -> list[Mol]:
    """RDKit Mol objects from smiles_list."""
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


@pytest.fixture(scope="session")
def single_mol() -> Mol:
    """Single molecule (phenol) for focused tests."""
    mol = Chem.MolFromSmiles("c1ccc(O)cc1")
    assert mol is not None
    return mol


@pytest.fixture(scope="session")
def benzene_mol() -> Mol:
    """Benzene molecule – simple aromatic for nitrogen walk tests."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None
    return mol


# ---------------------------------------------------------------------------
# Regression targets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def regression_targets(testing_df) -> np.ndarray:
    """First 50 regression targets from testing_data.csv."""
    return testing_df["Regression"].values[:50].astype(float)


@pytest.fixture(scope="session")
def small_regression_targets(testing_df) -> np.ndarray:
    """First 10 regression targets from testing_data.csv."""
    return testing_df["Regression"].values[:10].astype(float)


# ---------------------------------------------------------------------------
# LIME fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lime_desc() -> LIME:
    """LIME instance configured for descriptor-based analysis."""
    return LIME(
        descriptor_set=None,
        fingerprint_params=None,
        scale_coeff=True,
        use_fingerprints=False,
    )


@pytest.fixture()
def lime_ecfp() -> LIME:
    """LIME instance configured for ECFP-based analysis."""
    return LIME(
        descriptor_set=None,
        fingerprint_params=None,
        scale_coeff=True,
        use_fingerprints=True,
    )


# ---------------------------------------------------------------------------
# AnalogueGenerator fixture (not needed, it's a classmethod-only class)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MatchaExplainer fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_explainer() -> MatchaExplainer:
    """MatchaExplainer with default parameters."""
    return MatchaExplainer()
