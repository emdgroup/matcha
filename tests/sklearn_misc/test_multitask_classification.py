"""Test multi-task classification (num_endpoints > 1).

Model: MLPClassifier (tabular)
Feature: ``num_endpoints=2`` – the model should predict two binary targets.

Runtime optimisation: the model is fitted once via a fixture.
"""

import numpy as np
import pytest

from matcha.sklearn.tabular import MLPClassifier


@pytest.fixture()
def model_kwargs():
    return dict(
        hidden_dims=[8],
        feature_list=["ECFP"],
        num_endpoints=2,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


@pytest.fixture()
def fitted_model(mol_list, multitask_classification_y, model_kwargs):
    """Fit the MLPClassifier once; reuse for all prediction tests."""
    model = MLPClassifier(**model_kwargs)
    model.fit(mol_list, multitask_classification_y)
    return model


class TestMultitaskClassification:
    def test_predict_shape_is_multitask(self, fitted_model, mol_list):
        preds = fitted_model.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(mol_list), 2)

    def test_predict_values_are_binary(self, fitted_model, mol_list):
        preds = fitted_model.predict(mol_list)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0.0, 1.0})

    def test_predict_proba_shape_is_multitask(self, fitted_model, mol_list):
        proba = fitted_model.predict_proba(mol_list)
        assert proba.shape == (len(mol_list), 2)

    def test_predict_proba_in_0_1_range(self, fitted_model, mol_list):
        proba = fitted_model.predict_proba(mol_list)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
