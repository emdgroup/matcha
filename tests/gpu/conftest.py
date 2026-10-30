"""Shared pytest fixtures for GPU sklearn model tests.

Tests a subset of architectures (Chemprop, RoFormer, GIN, SNN) on GPU.

Fixtures provided:
* Raw data (molecules + labels from ``testing_data.csv``).
* ``regressor_cls`` / ``classifier_cls`` – parametrized over selected model classes.
* ``fitted_regressor`` / ``fitted_classifier`` – ready-to-predict model instances.
"""

import os
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

# ── CLM ──────────────────────────────────────────────────────────────────
from matcha.sklearn.clm import (
    RoFormerClassifier,
    RoFormerRegressor,
)

# ── Graph (2-D) ─────────────────────────────────────────────────────────
from matcha.sklearn.graph import (
    ChempropClassifier,
    ChempropRegressor,
    GINClassifier,
    GINRegressor,
)

# ── Tabular ─────────────────────────────────────────────────────────────
from matcha.sklearn.tabular import (
    SNNClassifier,
    SNNRegressor,
)

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
    """Regression labels (N, 1) for the first 30 compounds."""
    return testing_df["Regression"].values[:30].reshape(-1, 1).astype(np.float64)


@pytest.fixture(scope="session")
def classification_y(testing_df) -> np.ndarray:
    """Binary classification labels (N, 1) for the first 30 compounds."""
    return testing_df["Classification"].values[:30].reshape(-1, 1).astype(np.float64)


# =========================================================================
# Shared training kwargs fragments (GPU)
# =========================================================================

_BASE_TRAIN = dict(
    num_epochs=1,
    batch_size=32,
    accelerator="gpu",
    devices=1,
    early_stopping=False,
    stochastic_weight_averaging=False,
)

_GRAPH_PE_OFF = dict(
    rwse_k=0,
    laplacian_k=0,
    elstatic_k=0,
    distmat_k=0,
    rrwp_k=0,
    num_virtual_nodes=0,
)

# =========================================================================
# Per-architecture minimal kwargs  (class → kwargs mapping)
# =========================================================================

_ARCH_KWARGS: dict[type, dict] = {}

# ── CLM ─────────────────────────────────────────────────────────────────
_CLM_SHARED = {**_BASE_TRAIN, "num_augmentations": 1, "max_length": 100}

_ARCH_KWARGS[RoFormerClassifier] = {
    **_CLM_SHARED,
    "enc_hidden_dim": 32,
    "enc_expansion_dim": 64,
    "enc_num_heads": 4,
    "enc_num_layers": 1,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[RoFormerRegressor] = {**_ARCH_KWARGS[RoFormerClassifier]}

# ── Graph (2-D) ─────────────────────────────────────────────────────────
_GRAPH_SHARED = {**_BASE_TRAIN, **_GRAPH_PE_OFF}

_ARCH_KWARGS[GINClassifier] = {
    **_GRAPH_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[GINRegressor] = {**_ARCH_KWARGS[GINClassifier]}

_ARCH_KWARGS[ChempropClassifier] = {
    **_BASE_TRAIN,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "pred_hidden_dim": 32,
    "pred_num_layers": 1,
    "feature_list": None,
}
_ARCH_KWARGS[ChempropRegressor] = {**_ARCH_KWARGS[ChempropClassifier]}

# ── Tabular ─────────────────────────────────────────────────────────────
_TAB_SHARED = {
    **_BASE_TRAIN,
    "feature_list": ["ECFP"],
}

_ARCH_KWARGS[SNNClassifier] = {
    **_TAB_SHARED,
    "hidden_dims": [32],
    "num_parallel": 2,
}
_ARCH_KWARGS[SNNRegressor] = {**_ARCH_KWARGS[SNNClassifier]}

# =========================================================================
# Class lists (GPU subset)
# =========================================================================

_REGRESSOR_CLASSES = [
    # CLM
    RoFormerRegressor,
    # Graph
    GINRegressor,
    ChempropRegressor,
    # Tabular
    SNNRegressor,
]

_CLASSIFIER_CLASSES = [
    # CLM
    RoFormerClassifier,
    # Graph
    GINClassifier,
    ChempropClassifier,
    # Tabular
    SNNClassifier,
]


# =========================================================================
# Parametrized model-class fixtures
# =========================================================================


@pytest.fixture(params=_REGRESSOR_CLASSES, ids=lambda c: c.__name__)
def regressor_cls(request):
    """Yield one regressor class per parametrized run."""
    return request.param


@pytest.fixture(params=_CLASSIFIER_CLASSES, ids=lambda c: c.__name__)
def classifier_cls(request):
    """Yield one classifier class per parametrized run."""
    return request.param


@pytest.fixture()
def arch_kwargs(request):
    """Return the minimal kwargs dict for the current model class.

    Works with both ``regressor_cls`` and ``classifier_cls`` fixtures;
    the test just needs to pass the class to this fixture's inner helper.
    """
    return _ARCH_KWARGS


@pytest.fixture()
def fitted_regressor(regressor_cls, mol_list, regression_y):
    """Instantiate, fit, and return a regressor on the toy data."""
    kwargs = _ARCH_KWARGS[regressor_cls]
    model = regressor_cls(**kwargs)
    model.fit(mol_list, regression_y)
    return model


@pytest.fixture()
def fitted_classifier(classifier_cls, mol_list, classification_y):
    """Instantiate, fit, and return a classifier on the toy data."""
    kwargs = _ARCH_KWARGS[classifier_cls]
    model = classifier_cls(**kwargs)
    model.fit(mol_list, classification_y)
    return model
