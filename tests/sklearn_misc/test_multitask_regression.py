"""Test multi-task regression (num_endpoints > 1).

Model: GPSRegressor (graph)
Feature: ``num_endpoints=2`` – the model should predict two targets at once.

Runtime optimisation: the model is fitted once via a fixture.
"""

import numpy as np
import pytest

from matcha.sklearn.graph import GPSRegressor


@pytest.fixture()
def model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_atom_hidden_dim=8,
        enc_num_heads=4,
        enc_expansion_k=2,
        pred_hidden_dims=[8],
        rwse_k=0,
        laplacian_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
        num_virtual_nodes=0,
        num_endpoints=2,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


@pytest.fixture()
def fitted_model(mol_list, multitask_regression_y, model_kwargs):
    """Fit the GPSRegressor once; reuse for all predictions tests."""
    model = GPSRegressor(**model_kwargs)
    model.fit(mol_list, multitask_regression_y)
    return model


class TestMultitaskRegression:
    def test_predict_shape_is_multitask(self, fitted_model, mol_list):
        preds = fitted_model.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(mol_list), 2)

    def test_predict_values_are_finite(self, fitted_model, mol_list):
        preds = fitted_model.predict(mol_list)
        assert np.all(np.isfinite(preds))
