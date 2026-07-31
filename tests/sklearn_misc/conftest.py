"""Shared pytest fixtures for sklearn miscellaneous feature tests.

Provides molecule lists, label arrays, model factories, and pre-fitted
ensemble / single-model fixtures used across all test modules in this
folder.  Every fixture that needs ``testing_data.csv`` goes here so
individual test files stay focused on the feature under test.
"""

import os

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.sklearn.tabular import MLPClassifier, MLPRegressor
from matcha.sklearn.graph import GINClassifier, GINRegressor
from matcha.sklearn.graph import ChempropClassifier, ChempropRegressor
from matcha.sklearn.clm import CNNClassifier, CNNRegressor


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


@pytest.fixture(scope="session")
def multitask_regression_y(regression_y) -> np.ndarray:
    """Multi-task regression labels (N, 2) – column 0 is the original
    regression target, column 1 is a noisy copy."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, size=regression_y.shape)
    return np.hstack([regression_y, regression_y + noise])


@pytest.fixture(scope="session")
def multitask_classification_y(classification_y) -> np.ndarray:
    """Multi-task binary classification labels (N, 2) – column 0 is the
    original classification target, column 1 is a flipped copy."""
    flipped = 1.0 - classification_y
    return np.hstack([classification_y, flipped])


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

N_MODELS = 2  # keep ensembles small for speed


# =========================================================================
# Model factory helpers  (importable by test modules)
# =========================================================================


def make_mlp_regressor():
    return MLPRegressor(
        hidden_dims=[32],
        feature_list=["ECFP"],
        **BASE_TRAIN,
    )


def make_mlp_classifier():
    return MLPClassifier(
        hidden_dims=[32],
        feature_list=["ECFP"],
        **BASE_TRAIN,
    )


def make_gin_regressor():
    return GINRegressor(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dims=[32],
        **GRAPH_PE_OFF,
        **BASE_TRAIN,
    )


def make_gin_classifier():
    return GINClassifier(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dims=[32],
        **GRAPH_PE_OFF,
        **BASE_TRAIN,
    )


def make_cnn_regressor(num_augmentations=1):
    return CNNRegressor(
        enc_hidden_dim=32,
        enc_kernel_dims=[3, 5],
        enc_num_heads=4,
        pred_hidden_dims=[32],
        num_augmentations=num_augmentations,
        max_length=100,
        **BASE_TRAIN,
    )


def make_cnn_classifier(num_augmentations=1):
    return CNNClassifier(
        enc_hidden_dim=32,
        enc_kernel_dims=[3, 5],
        enc_num_heads=4,
        pred_hidden_dims=[32],
        num_augmentations=num_augmentations,
        max_length=100,
        **BASE_TRAIN,
    )


def make_chemprop_regressor():
    return ChempropRegressor(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dim=32,
        pred_num_layers=1,
        feature_list=None,
        **BASE_TRAIN,
    )


def make_chemprop_classifier():
    return ChempropClassifier(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dim=32,
        pred_num_layers=1,
        feature_list=None,
        **BASE_TRAIN,
    )


# =========================================================================
# Parametrized factory lists (reusable by test modules)
# =========================================================================

REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLPRegressor"),
    pytest.param(make_gin_regressor, id="GINRegressor"),
    pytest.param(make_cnn_regressor, id="CNNRegressor"),
    pytest.param(make_chemprop_regressor, id="ChempropRegressor"),
]

CLASSIFIER_FACTORIES = [
    pytest.param(make_mlp_classifier, id="MLPClassifier"),
    pytest.param(make_gin_classifier, id="GINClassifier"),
    pytest.param(make_cnn_classifier, id="CNNClassifier"),
    pytest.param(make_chemprop_classifier, id="ChempropClassifier"),
]


@pytest.fixture(params=REGRESSOR_FACTORIES)
def regressor_factory(request):
    """Yield a callable that creates an unfitted regressor."""
    return request.param


@pytest.fixture(params=CLASSIFIER_FACTORIES)
def classifier_factory(request):
    """Yield a callable that creates an unfitted classifier."""
    return request.param
