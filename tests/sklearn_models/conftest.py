"""Shared pytest fixtures for sklearn model tests.

Every architecture (CLM, graph, graph3D, tabular) is tested for both
regression and classification via parametrized fixtures.

Fixtures provided:
* Raw data (molecules + labels from ``testing_data.csv``).
* ``regressor_cls`` / ``classifier_cls`` – parametrized over all model classes.
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
    CNNClassifier,
    CNNRegressor,
    RNNClassifier,
    RNNRegressor,
    RoFormerClassifier,
    RoFormerRegressor,
)

# ── Graph (2-D) ─────────────────────────────────────────────────────────
from matcha.sklearn.graph import (
    AttentiveFPClassifier,
    AttentiveFPRegressor,
    ChempropClassifier,
    ChempropRegressor,
    GatedGCNClassifier,
    GatedGCNRegressor,
    GINClassifier,
    GINRegressor,
    GPSClassifier,
    GPSRegressor,
    GTClassifier,
    GTRegressor,
)

# ── Graph 3-D ───────────────────────────────────────────────────────────
from matcha.sklearn.graph3d import (
    E3GNNClassifier,
    E3GNNRegressor,
    GPS3DClassifier,
    GPS3DRegressor,
    GT3DClassifier,
    GT3DRegressor,
)

# ── Tabular ─────────────────────────────────────────────────────────────
from matcha.sklearn.tabular import (
    MLPClassifier,
    MLPRegressor,
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
# Shared training kwargs fragments
# =========================================================================

_BASE_TRAIN = dict(
    num_epochs=1,
    batch_size=32,
    accelerator="cpu",
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

_ARCH_KWARGS[CNNClassifier] = {
    **_CLM_SHARED,
    "enc_hidden_dim": 32,
    "enc_kernel_dims": [3, 5],
    "enc_num_heads": 4,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[CNNRegressor] = {**_ARCH_KWARGS[CNNClassifier]}

_ARCH_KWARGS[RNNClassifier] = {
    **_CLM_SHARED,
    "enc_num_layers": 1,
    "enc_embedding_dim": 32,
    "enc_hidden_dim": 32,
    "enc_num_heads": 4,
    "enc_bidirectional": False,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[RNNRegressor] = {**_ARCH_KWARGS[RNNClassifier]}

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
    "enc_eps": 0.0,
    "enc_train_eps": True,
}
_ARCH_KWARGS[GINRegressor] = {**_ARCH_KWARGS[GINClassifier]}

_ARCH_KWARGS[AttentiveFPClassifier] = {
    **_GRAPH_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[AttentiveFPRegressor] = {**_ARCH_KWARGS[AttentiveFPClassifier]}

_ARCH_KWARGS[GatedGCNClassifier] = {
    **_GRAPH_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[GatedGCNRegressor] = {**_ARCH_KWARGS[GatedGCNClassifier]}

_ARCH_KWARGS[GPSClassifier] = {
    **_GRAPH_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "enc_num_heads": 4,
    "enc_expansion_k": 2,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[GPSRegressor] = {**_ARCH_KWARGS[GPSClassifier]}

_ARCH_KWARGS[GTClassifier] = {
    **_GRAPH_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "enc_num_heads": 4,
    "enc_expansion_k": 2,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[GTRegressor] = {**_ARCH_KWARGS[GTClassifier]}

_ARCH_KWARGS[ChempropClassifier] = {
    **_BASE_TRAIN,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "pred_hidden_dim": 32,
    "pred_num_layers": 1,
    "feature_list": None,
}
_ARCH_KWARGS[ChempropRegressor] = {**_ARCH_KWARGS[ChempropClassifier]}

# ── Graph 3-D ───────────────────────────────────────────────────────────
_GRAPH3D_SHARED = {**_BASE_TRAIN, **_GRAPH_PE_OFF}

_ARCH_KWARGS[E3GNNClassifier] = {
    **_GRAPH3D_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "enc_m_dim": 8,
    "enc_fourier_features": 2,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[E3GNNRegressor] = {**_ARCH_KWARGS[E3GNNClassifier]}

_ARCH_KWARGS[GPS3DClassifier] = {
    **_GRAPH3D_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "enc_num_heads": 4,
    "enc_expansion_k": 2,
    "enc_num_kernels": 2,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[GPS3DRegressor] = {**_ARCH_KWARGS[GPS3DClassifier]}

_ARCH_KWARGS[GT3DClassifier] = {
    **_GRAPH3D_SHARED,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "enc_num_heads": 4,
    "enc_expansion_k": 2,
    "enc_num_kernels": 2,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[GT3DRegressor] = {**_ARCH_KWARGS[GT3DClassifier]}

# ── Tabular ─────────────────────────────────────────────────────────────
_TAB_SHARED = {
    **_BASE_TRAIN,
    "feature_list": ["ECFP"],
}

_ARCH_KWARGS[MLPClassifier] = {
    **_TAB_SHARED,
    "hidden_dims": [32],
}
_ARCH_KWARGS[MLPRegressor] = {**_ARCH_KWARGS[MLPClassifier]}

_ARCH_KWARGS[SNNClassifier] = {
    **_TAB_SHARED,
    "hidden_dims": [32],
    "num_parallel": 2,
}
_ARCH_KWARGS[SNNRegressor] = {**_ARCH_KWARGS[SNNClassifier]}

# =========================================================================
# Class lists
# =========================================================================

_REGRESSOR_CLASSES = [
    # CLM
    CNNRegressor,
    RNNRegressor,
    RoFormerRegressor,
    # Graph
    GINRegressor,
    AttentiveFPRegressor,
    GatedGCNRegressor,
    GPSRegressor,
    GTRegressor,
    ChempropRegressor,
    # Graph 3-D
    E3GNNRegressor,
    GPS3DRegressor,
    GT3DRegressor,
    # Tabular
    MLPRegressor,
    SNNRegressor,
]

_CLASSIFIER_CLASSES = [
    # CLM
    CNNClassifier,
    RNNClassifier,
    RoFormerClassifier,
    # Graph
    GINClassifier,
    AttentiveFPClassifier,
    GatedGCNClassifier,
    GPSClassifier,
    GTClassifier,
    ChempropClassifier,
    # Graph 3-D
    E3GNNClassifier,
    GPS3DClassifier,
    GT3DClassifier,
    # Tabular
    MLPClassifier,
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
