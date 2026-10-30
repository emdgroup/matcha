"""Test Monte-Carlo dropout uncertainty estimation.

Model: RoFormerRegressor (CLM)
Feature: ``compute_uncertainty()`` – runs MC-dropout forward passes and
returns an uncertainty (std) array with the same leading dimension as
the input.
"""

import numpy as np
import pytest
from rdkit.Chem.rdchem import Mol

from matcha.sklearn.clm import RoFormerRegressor


@pytest.fixture()
def fitted_model(mol_list: list[Mol], regression_y):
    model = RoFormerRegressor(
        enc_hidden_dim=8,
        enc_expansion_dim=16,
        enc_num_heads=4,
        enc_num_layers=1,
        pred_hidden_dims=[8],
        num_augmentations=1,
        max_length=100,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )
    model.fit(mol_list, regression_y)
    return model


class TestMCDropout:
    def test_uncertainty_returns_ndarray(self, fitted_model, mol_list: list[Mol]):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert isinstance(unc, np.ndarray)

    def test_uncertainty_shape_matches_input(self, fitted_model, mol_list: list[Mol]):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert unc.shape[0] == len(mol_list)

    def test_uncertainty_values_are_non_negative(
        self, fitted_model, mol_list: list[Mol]
    ):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert np.all(unc >= 0.0)

    def test_uncertainty_values_are_finite(self, fitted_model, mol_list: list[Mol]):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert np.all(np.isfinite(unc))
