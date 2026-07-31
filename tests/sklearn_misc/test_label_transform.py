"""Test log10 label transform applied to a regression model.

Model: GINRegressor (graph)
Feature: ``label_transform_map="log10"`` – the datamodule should apply
a log10 transform to labels before training and invert it on predict.

The regression targets in testing_data.csv are already negative (log-scale
values), so we shift them to be strictly positive before applying log10.

Runtime optimisation: the model is fitted once via a fixture.
"""

import numpy as np
import pytest

from matcha.sklearn.graph import GINRegressor


@pytest.fixture()
def positive_regression_y(regression_y) -> np.ndarray:
    """Shift regression labels so all values are strictly positive
    (required for a log10 transform)."""
    shifted = regression_y - regression_y.min() + 1.0
    return shifted


@pytest.fixture()
def model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dims=[32],
        rwse_k=0,
        laplacian_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
        num_virtual_nodes=0,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
        label_transform_map="log10",
    )


@pytest.fixture()
def fitted_model(mol_list, positive_regression_y, model_kwargs):
    """Fit the GINRegressor once; reuse for all prediction tests."""
    model = GINRegressor(**model_kwargs)
    model.fit(mol_list, positive_regression_y)
    return model


class TestLog10Transform:
    def test_predict_returns_finite(self, fitted_model, mol_list):
        preds = fitted_model.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert np.all(np.isfinite(preds))
