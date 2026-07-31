"""Tests for all sklearn *regressor* architectures: fit and predict on toy data.

Each test is parametrized over every regressor class (CLM, graph, graph3D,
tabular) so that a failure clearly identifies which architecture broke.

All model classes, kwargs, and parametrized fixtures are defined in conftest.py
and auto-discovered by pytest – no explicit import needed.
"""

import numpy as np
from rdkit.Chem.rdchem import Mol


# ---------------------------------------------------------------------------
# fit tests
# ---------------------------------------------------------------------------


class TestRegressorFit:
    """Verify that fit() runs and produces a stored model."""

    def test_fit_completes(
        self, regressor_cls, mol_list: list[Mol], regression_y: np.ndarray, arch_kwargs
    ):
        kwargs = arch_kwargs[regressor_cls]
        model = regressor_cls(**kwargs)
        model.fit(mol_list, regression_y)

    def test_fit_stores_model(
        self, regressor_cls, mol_list: list[Mol], regression_y: np.ndarray, arch_kwargs
    ):
        kwargs = arch_kwargs[regressor_cls]
        model = regressor_cls(**kwargs)
        model.fit(mol_list, regression_y)
        assert model._model is not None


# ---------------------------------------------------------------------------
# predict tests
# ---------------------------------------------------------------------------


class TestRegressorPredict:
    """Verify that predict() returns well-formed output."""

    def test_predict_returns_ndarray(self, fitted_regressor, mol_list: list[Mol]):
        preds = fitted_regressor.predict(mol_list)
        assert isinstance(preds, np.ndarray)

    def test_predict_shape_matches_input(self, fitted_regressor, mol_list: list[Mol]):
        preds = fitted_regressor.predict(mol_list)
        assert preds.shape[0] == len(mol_list)

    def test_predict_values_are_finite(self, fitted_regressor, mol_list: list[Mol]):
        preds = fitted_regressor.predict(mol_list)
        assert np.all(np.isfinite(preds))
