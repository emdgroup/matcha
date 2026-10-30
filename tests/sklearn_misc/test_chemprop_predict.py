"""End-to-end predict smoke tests for the Chemprop wrapper.

Exercises the shared direct-loop predict path (post-issue-89 stage 2) with
and without extra descriptor features. This covers the acceptance criterion
that predict succeeds on Chemprop-based estimators in both configurations.
"""

import numpy as np
import pytest
import torch
from rdkit.Chem.rdchem import Mol

from matcha.sklearn.graph import ChempropRegressor


def _make_regressor(**overrides) -> ChempropRegressor:
    """Build a small ChempropRegressor with sensible defaults for tests."""
    kwargs = {
        "enc_num_layers": 1,
        "enc_atom_hidden_dim": 32,
        "pred_hidden_dim": 32,
        "pred_num_layers": 1,
        "num_epochs": 1,
        "batch_size": 8,
        "accelerator": "cpu",
        "devices": 1,
        "early_stopping": False,
        "stochastic_weight_averaging": False,
    }
    kwargs.update(overrides)
    return ChempropRegressor(**kwargs)


@pytest.fixture
def fitted_chemprop_no_features(mol_list: list[Mol], regression_y: np.ndarray):
    model = _make_regressor(feature_list=None)
    model.fit(mol_list, regression_y)
    return model


@pytest.fixture
def fitted_chemprop_with_features(mol_list: list[Mol], regression_y: np.ndarray):
    model = _make_regressor(feature_list=["estate"])
    model.fit(mol_list, regression_y)
    return model


class TestChempropPredict:
    """Direct-loop predict must succeed on Chemprop with and without features."""

    def test_predict_without_features(
        self, fitted_chemprop_no_features, mol_list: list[Mol]
    ):
        preds = fitted_chemprop_no_features.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert np.all(np.isfinite(preds))

    def test_predict_with_features(
        self, fitted_chemprop_with_features, mol_list: list[Mol]
    ):
        preds = fitted_chemprop_with_features.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert np.all(np.isfinite(preds))

    def test_inner_predict_returns_tensor(
        self, fitted_chemprop_no_features, mol_list: list[Mol]
    ):
        preds = fitted_chemprop_no_features._inner_predict(mol_list)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape[0] == len(mol_list)
