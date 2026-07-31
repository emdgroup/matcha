"""Test UncertaintyManager through the sklearn API.

Model: MLPRegressor (tabular)
Exercises: compute (MC-dropout uncertainty), create_calibrator, calibrator
    property, params property, manual calibration flow.
"""

import numpy as np
import pytest
from rdkit.Chem.rdchem import Mol

from matcha.sklearn.tabular import MLPRegressor
from matcha.sklearn.managers import UncertaintyManager


@pytest.fixture()
def model_kwargs():
    return dict(
        hidden_dims=[32],
        feature_list=["ECFP"],
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


@pytest.fixture()
def fitted_model(mol_list: list[Mol], regression_y, model_kwargs):
    model = MLPRegressor(**model_kwargs)
    model.fit(mol_list, regression_y)
    return model


class TestUncertaintyManagerInit:
    """Tests for initial state of UncertaintyManager."""

    def test_calibrator_is_none_by_default(self):
        mgr = UncertaintyManager()
        assert mgr.calibrator is None

    def test_params_is_none_by_default(self):
        mgr = UncertaintyManager()
        assert mgr.params is None


class TestUncertaintyManagerCompute:
    """Tests for compute_uncertainty via MC dropout."""

    def test_uncertainty_returns_ndarray(self, fitted_model, mol_list):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert isinstance(unc, np.ndarray)

    def test_uncertainty_shape_matches_input(self, fitted_model, mol_list):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert unc.shape[0] == len(mol_list)

    def test_uncertainty_has_correct_num_tasks(self, fitted_model, mol_list):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        # Single-task regression → second dimension should be 1
        assert unc.shape[1] == 1

    def test_uncertainty_values_are_non_negative(self, fitted_model, mol_list):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert np.all(unc >= 0.0)

    def test_uncertainty_values_are_finite(self, fitted_model, mol_list):
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert np.all(np.isfinite(unc))


class TestUncertaintyManagerCreateCalibrator:
    """Tests for create_calibrator and manual calibration flow."""

    def test_create_calibrator_sets_instance(self):
        mgr = UncertaintyManager()
        mgr.create_calibrator("icp_regression", {"confidence_alpha": 0.2})
        assert mgr.calibrator is not None

    def test_create_calibrator_params_populated(self):
        mgr = UncertaintyManager()
        mgr.create_calibrator("icp_regression", {"confidence_alpha": 0.2})
        assert mgr.params is not None
        assert mgr.params.calibrator_type == "icp_regression"

    def test_manual_calibration_flow(self, fitted_model, mol_list, regression_y):
        """Manually perform the calibration steps that calibrate() does,
        but compute raw std *before* creating the calibrator to avoid the
        ordering bug."""
        mgr = fitted_model._uncertainty_manager

        # 1. Compute raw uncertainty (no calibrator yet)
        assert mgr.calibrator is None
        raw_std = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        preds = fitted_model.predict(mol_list)

        # 2. Create the calibrator
        mgr.create_calibrator("icp_regression", {"confidence_alpha": 0.2})

        # 3. Fit the calibrator
        mgr.calibrator.fit(regression_y, preds, raw_std)

        # 4. Now compute_uncertainty should apply the calibrator
        calibrated_std = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert isinstance(calibrated_std, np.ndarray)
        assert calibrated_std.shape[0] == len(mol_list)
        assert np.all(calibrated_std >= 0.0)
        assert np.all(np.isfinite(calibrated_std))


class TestUncertaintyManagerCalibrateEndToEnd:
    """Tests for calibrate_uncertainty end-to-end through the sklearn API."""

    def test_calibrate_sets_calibrator(self, fitted_model, mol_list, regression_y):
        fitted_model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            num_iterations=3,
            algorithm="icp_regression",
        )
        assert fitted_model._uncertainty_manager.calibrator is not None

    def test_calibrate_params_populated(self, fitted_model, mol_list, regression_y):
        fitted_model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            num_iterations=3,
            algorithm="icp_regression",
        )
        assert fitted_model._uncertainty_manager.params is not None

    def test_calibrated_uncertainty_returns_ndarray(
        self, fitted_model, mol_list, regression_y
    ):
        fitted_model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            num_iterations=3,
            algorithm="icp_regression",
        )
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert isinstance(unc, np.ndarray)
        assert unc.shape[0] == len(mol_list)

    def test_calibrated_uncertainty_is_non_negative(
        self, fitted_model, mol_list, regression_y
    ):
        fitted_model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            num_iterations=3,
            algorithm="icp_regression",
        )
        unc = fitted_model.compute_uncertainty(mol_list, num_iterations=3)
        assert np.all(unc >= 0.0)
