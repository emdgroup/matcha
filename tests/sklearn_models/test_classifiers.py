"""Tests for all sklearn *classifier* architectures: fit, predict, predict_proba.

Each test is parametrized over every classifier class (CLM, graph, graph3D,
tabular) so that a failure clearly identifies which architecture broke.

All model classes, kwargs, and parametrized fixtures are defined in conftest.py
and auto-discovered by pytest – no explicit import needed.
"""

import numpy as np
from rdkit.Chem.rdchem import Mol


# ---------------------------------------------------------------------------
# fit tests
# ---------------------------------------------------------------------------


class TestClassifierFit:
    """Verify that fit() runs and produces a stored model."""

    def test_fit_completes(
        self,
        classifier_cls,
        mol_list: list[Mol],
        classification_y: np.ndarray,
        arch_kwargs,
    ):
        kwargs = arch_kwargs[classifier_cls]
        model = classifier_cls(**kwargs)
        model.fit(mol_list, classification_y)

    def test_fit_stores_model(
        self,
        classifier_cls,
        mol_list: list[Mol],
        classification_y: np.ndarray,
        arch_kwargs,
    ):
        kwargs = arch_kwargs[classifier_cls]
        model = classifier_cls(**kwargs)
        model.fit(mol_list, classification_y)
        assert model._model is not None


# ---------------------------------------------------------------------------
# predict tests
# ---------------------------------------------------------------------------


class TestClassifierPredict:
    """Verify that predict() returns well-formed binary output."""

    def test_predict_returns_ndarray(self, fitted_classifier, mol_list: list[Mol]):
        preds = fitted_classifier.predict(mol_list)
        assert isinstance(preds, np.ndarray)

    def test_predict_shape_matches_input(self, fitted_classifier, mol_list: list[Mol]):
        preds = fitted_classifier.predict(mol_list)
        assert preds.shape[0] == len(mol_list)

    def test_predict_values_are_binary(self, fitted_classifier, mol_list: list[Mol]):
        preds = fitted_classifier.predict(mol_list)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# predict_proba tests
# ---------------------------------------------------------------------------


class TestClassifierPredictProba:
    """Verify that predict_proba() returns valid probabilities."""

    def test_predict_proba_returns_ndarray(
        self, fitted_classifier, mol_list: list[Mol]
    ):
        proba = fitted_classifier.predict_proba(mol_list)
        assert isinstance(proba, np.ndarray)

    def test_predict_proba_shape_matches_input(
        self, fitted_classifier, mol_list: list[Mol]
    ):
        proba = fitted_classifier.predict_proba(mol_list)
        assert proba.shape[0] == len(mol_list)

    def test_predict_proba_in_0_1_range(self, fitted_classifier, mol_list: list[Mol]):
        proba = fitted_classifier.predict_proba(mol_list)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
