"""Test UncertaintyManager through the sklearn API.

Model: MLPRegressor (tabular)
Exercises: compute (MC-dropout and MVE uncertainty), create_calibrator,
    calibrator property, params property, manual calibration flow.
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


@pytest.fixture()
def fitted_mve_model(mol_list: list[Mol], regression_y, model_kwargs):
    model = MLPRegressor(**model_kwargs, uncertainty="mve", loss_fn="beta-nll")
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


class TestUncertaintyManagerComputeMVE:
    """Tests for the MVE branch of ``UncertaintyManager.compute``."""

    def test_mve_uncertainty_returns_ndarray(self, fitted_mve_model, mol_list):
        unc = fitted_mve_model.compute_uncertainty(mol_list)
        assert isinstance(unc, np.ndarray)

    def test_mve_uncertainty_shape(self, fitted_mve_model, mol_list):
        unc = fitted_mve_model.compute_uncertainty(mol_list)
        assert unc.shape == (len(mol_list), 1)

    def test_mve_uncertainty_is_finite(self, fitted_mve_model, mol_list):
        unc = fitted_mve_model.compute_uncertainty(mol_list)
        assert np.all(np.isfinite(unc))

    def test_mve_uncertainty_is_non_negative(self, fitted_mve_model, mol_list):
        unc = fitted_mve_model.compute_uncertainty(mol_list)
        assert np.all(unc >= 0.0)

    def test_mve_uncertainty_matches_manual_variance_transform(
        self, fitted_mve_model, mol_list
    ):
        """``_compute_mve`` must invert the target scaler's variance transform.

        The scaled-space std returned by ``predict_variance_step`` should be
        multiplied by ``y_scaler.scale_`` to land in the original label space.
        """
        _, log_var_scaled = fitted_mve_model._inner_predict_variance(mol_list)
        scale = fitted_mve_model.datamodule._y_scaler.scale_
        expected_std = np.sqrt(np.exp(log_var_scaled.numpy()) * (scale**2))

        unc = fitted_mve_model.compute_uncertainty(mol_list)
        np.testing.assert_allclose(unc, expected_std, rtol=1e-5)

    def test_mve_bypasses_mc_dropout_iterations(self, fitted_mve_model, mol_list):
        """MVE path should not depend on ``num_iterations`` — a single forward
        pass on the same input must be deterministic between calls (dropout
        forced off in ``predict_variance_step``)."""
        unc_a = fitted_mve_model.compute_uncertainty(mol_list, num_iterations=1)
        unc_b = fitted_mve_model.compute_uncertainty(mol_list, num_iterations=50)
        np.testing.assert_allclose(unc_a, unc_b, rtol=1e-5)

    def test_mve_with_calibrator_applies_calibration(
        self, fitted_mve_model, mol_list, regression_y
    ):
        mgr = fitted_mve_model._uncertainty_manager
        raw_std = fitted_mve_model.compute_uncertainty(mol_list)
        preds = fitted_mve_model.predict(mol_list)

        mgr.create_calibrator("icp_regression", {"confidence_alpha": 0.2})
        mgr.calibrator.fit(regression_y, preds, raw_std)

        calibrated_std = fitted_mve_model.compute_uncertainty(mol_list)
        assert calibrated_std.shape == raw_std.shape
        assert np.all(calibrated_std >= 0.0)
        assert np.all(np.isfinite(calibrated_std))

    def test_inner_predict_variance_raises_on_non_mve_model(
        self, fitted_model, mol_list
    ):
        with pytest.raises(RuntimeError, match="uncertainty='mve'"):
            fitted_model._inner_predict_variance(mol_list)


class TestUncertaintyManagerDispatch:
    """The compute() dispatch selects a branch from the model's uncertainty_method."""

    class _StubModel:
        """Minimal model_instance stub — only ``_model.uncertainty_method`` is read."""

        def __init__(self, method: str):
            self._model = type("_M", (), {"uncertainty_method": method})()

    @pytest.mark.parametrize("method", ["mc-dropout", "mve"])
    def test_dispatch_selects_correct_branch(self, monkeypatch, method):
        mgr = UncertaintyManager()
        calls = {"mc-dropout": 0, "mve": 0}

        def fake_mc(self, *args, **kwargs):
            calls["mc-dropout"] += 1
            return np.zeros((1, 1))

        def fake_mve(self, *args, **kwargs):
            calls["mve"] += 1
            return np.zeros((1, 1))

        monkeypatch.setattr(UncertaintyManager, "_compute_mc_dropout", fake_mc)
        monkeypatch.setattr(UncertaintyManager, "_compute_mve", fake_mve)

        mgr.compute(self._StubModel(method), x=None)
        assert calls[method] == 1
        other = "mve" if method == "mc-dropout" else "mc-dropout"
        assert calls[other] == 0

    def test_dispatch_raises_on_unknown_method(self, monkeypatch):
        mgr = UncertaintyManager()
        # Bypass the concrete branches so an unknown method reliably hits the fallback.
        monkeypatch.setattr(
            UncertaintyManager,
            "_compute_mc_dropout",
            lambda self, *a, **kw: np.zeros((1, 1)),
        )
        monkeypatch.setattr(
            UncertaintyManager,
            "_compute_mve",
            lambda self, *a, **kw: np.zeros((1, 1)),
        )
        with pytest.raises(NotImplementedError, match="Unknown uncertainty method"):
            mgr.compute(self._StubModel("deep-ensemble"), x=None)
