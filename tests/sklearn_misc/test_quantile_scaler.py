"""Test quantile scaler for label standardisation.

Model: RNNRegressor (CLM)
Feature: ``scaler_type="quantile"`` – uses QuantileTransformer instead of
StandardScaler for label normalisation before training.

Runtime optimisation: the model is fitted once via a fixture.
"""

import numpy as np
import pytest

from matcha.sklearn.clm import RNNRegressor


@pytest.fixture()
def model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_embedding_dim=8,
        enc_hidden_dim=8,
        enc_num_heads=4,
        enc_bidirectional=False,
        pred_hidden_dims=[8],
        num_augmentations=1,
        max_length=100,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
        scaler_type="quantile",
    )


@pytest.fixture()
def fitted_model(mol_list, regression_y, model_kwargs):
    """Fit the RNNRegressor once; reuse for all prediction tests."""
    model = RNNRegressor(**model_kwargs)
    model.fit(mol_list, regression_y)
    return model


class TestQuantileScaler:
    def test_predict_returns_finite(self, fitted_model, mol_list):
        preds = fitted_model.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert np.all(np.isfinite(preds))
