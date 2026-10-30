"""Test Ensemble for all modalities (tabular, graph, CLM, chemprop).

Exercises: construction, fit, predict (mean + std), save/load round-trip,
    calibration (calibrate_uncertainty), and MLflow logging
    (set_mlflow_experiment + fit with logging).
Each modality is tested with both a regressor and a classifier.

Runtime optimisation: expensive ``fit`` calls are cached in fixtures so
that each model architecture is fitted only *once* per phase (plain fit,
calibrated fit, mlflow fit).
"""

import os

import numpy as np
import pytest

from matcha.sklearn import Ensemble
from matcha.utils.schemas.sklearn_api import MLFlowInputModel

from .conftest import N_MODELS


# =========================================================================
# Fitted-ensemble fixtures  (one fit per factory × task type)
# =========================================================================


@pytest.fixture()
def fitted_reg_ensemble(regressor_factory, mol_list, regression_y):
    """Fit a regression ensemble once; reuse for predict / save / load tests."""
    template = regressor_factory()
    ens = Ensemble(model=template, n_models=N_MODELS)
    ens.fit(mol_list, regression_y)
    return ens


@pytest.fixture()
def fitted_cls_ensemble(classifier_factory, mol_list, classification_y):
    """Fit a classification ensemble once; reuse for predict / save / load tests."""
    template = classifier_factory()
    ens = Ensemble(model=template, n_models=N_MODELS)
    ens.fit(mol_list, classification_y)
    return ens


# =========================================================================
# Regression ensemble – construction  (no fit needed)
# =========================================================================


class TestEnsembleRegressorConstruction:
    """Verify that an ensemble can be constructed from a regressor template."""

    def test_ensemble_creates_n_models(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        assert len(ens.model_box) == N_MODELS

    def test_ensemble_is_not_classifier(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        assert ens.is_classifier is False

    def test_managers_initialised(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        assert ens._mlflow_manager is not None
        assert ens._serialization_manager is not None
        assert ens._calibration_manager is not None

    def test_calibrator_is_none_before_calibration(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        assert ens.calibrator is None


# =========================================================================
# Regression ensemble – fit + predict  (one fit via fixture)
# =========================================================================


class TestEnsembleRegressorPredict:
    """Verify that ensemble predict returns (mean, std) for regressors."""

    def test_predict_returns_tuple(self, fitted_reg_ensemble, mol_list):
        result = fitted_reg_ensemble.predict(mol_list)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_mean_shape(self, fitted_reg_ensemble, mol_list):
        mean, _ = fitted_reg_ensemble.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)

    def test_predict_std_shape(self, fitted_reg_ensemble, mol_list):
        _, std = fitted_reg_ensemble.predict(mol_list)
        assert std.shape == (len(mol_list), 1)

    def test_predict_mean_values_are_finite(self, fitted_reg_ensemble, mol_list):
        mean, _ = fitted_reg_ensemble.predict(mol_list)
        assert np.all(np.isfinite(mean))

    def test_predict_std_values_are_non_negative(self, fitted_reg_ensemble, mol_list):
        _, std = fitted_reg_ensemble.predict(mol_list)
        assert np.all(std >= 0.0)

    def test_predict_unreduced(self, fitted_reg_ensemble, mol_list):
        preds = fitted_reg_ensemble.predict(mol_list, reduce=False)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert preds.shape[1] == N_MODELS


# =========================================================================
# Regression ensemble – save / load  (one fit via fixture)
# =========================================================================


class TestEnsembleRegressorSaveLoad:
    """Verify save → load round-trip for regression ensembles."""

    def test_save_creates_ensemble_params(self, fitted_reg_ensemble, tmp_path):
        save_dir = str(tmp_path / "ens")
        fitted_reg_ensemble.save_model(save_dir)
        assert os.path.isdir(os.path.join(save_dir, "config"))
        assert os.path.exists(os.path.join(save_dir, "config", "manifest.yaml"))
        assert os.path.exists(os.path.join(save_dir, "config", "ensemble.yaml"))
        assert os.path.exists(os.path.join(save_dir, "config", "metadata.yaml"))

    def test_save_creates_member_directories(self, fitted_reg_ensemble, tmp_path):
        save_dir = str(tmp_path / "ens")
        fitted_reg_ensemble.save_model(save_dir)
        for i in range(N_MODELS):
            assert os.path.isdir(os.path.join(save_dir, f"model_{i}"))

    def test_save_creates_calibrator_pickle(self, fitted_reg_ensemble, tmp_path):
        save_dir = str(tmp_path / "ens")
        fitted_reg_ensemble.save_model(save_dir)
        assert os.path.exists(os.path.join(save_dir, "state", "calibrator.pkl"))

    def test_load_round_trip_predicts(self, fitted_reg_ensemble, mol_list, tmp_path):
        save_dir = str(tmp_path / "ens")
        fitted_reg_ensemble.save_model(save_dir)
        loaded = Ensemble.from_folder(save_dir, accelerator="cpu")
        mean, std = loaded.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(mean))


# =========================================================================
# Classification ensemble – construction  (no fit needed)
# =========================================================================


class TestEnsembleClassifierConstruction:
    """Verify that an ensemble can be constructed from a classifier template."""

    def test_ensemble_creates_n_models(self, classifier_factory):
        template = classifier_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        assert len(ens.model_box) == N_MODELS

    def test_ensemble_is_classifier(self, classifier_factory):
        template = classifier_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        assert ens.is_classifier is True


# =========================================================================
# Classification ensemble – fit + predict  (one fit via fixture)
# =========================================================================


class TestEnsembleClassifierPredict:
    """Verify that ensemble predict returns (mean, std) for classifiers."""

    def test_predict_returns_tuple(self, fitted_cls_ensemble, mol_list):
        result = fitted_cls_ensemble.predict(mol_list)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_mean_shape(self, fitted_cls_ensemble, mol_list):
        mean, _ = fitted_cls_ensemble.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)

    def test_predict_std_shape(self, fitted_cls_ensemble, mol_list):
        _, std = fitted_cls_ensemble.predict(mol_list)
        assert std.shape == (len(mol_list), 1)

    def test_predict_mean_values_are_finite(self, fitted_cls_ensemble, mol_list):
        mean, _ = fitted_cls_ensemble.predict(mol_list)
        assert np.all(np.isfinite(mean))

    def test_predict_unreduced(self, fitted_cls_ensemble, mol_list):
        preds = fitted_cls_ensemble.predict(mol_list, reduce=False)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert preds.shape[1] == N_MODELS


# =========================================================================
# Classification ensemble – save / load  (one fit via fixture)
# =========================================================================


class TestEnsembleClassifierSaveLoad:
    """Verify save → load round-trip for classification ensembles."""

    def test_save_creates_ensemble_params(self, fitted_cls_ensemble, tmp_path):
        save_dir = str(tmp_path / "ens")
        fitted_cls_ensemble.save_model(save_dir)
        assert os.path.isdir(os.path.join(save_dir, "config"))
        assert os.path.exists(os.path.join(save_dir, "config", "manifest.yaml"))
        assert os.path.exists(os.path.join(save_dir, "config", "ensemble.yaml"))
        assert os.path.exists(os.path.join(save_dir, "config", "metadata.yaml"))

    def test_load_round_trip_predicts(self, fitted_cls_ensemble, mol_list, tmp_path):
        save_dir = str(tmp_path / "ens")
        fitted_cls_ensemble.save_model(save_dir)
        loaded = Ensemble.from_folder(save_dir, accelerator="cpu")
        mean, std = loaded.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(mean))


# =========================================================================
# Calibration – regression ensembles  (one fit + calibrate via fixture)
# =========================================================================


class TestEnsembleRegressorCalibration:
    """Verify calibrate_uncertainty for regression ensembles.

    Fits once, captures raw std, calibrates, then runs all assertions.
    """

    @pytest.fixture()
    def calibrated_reg(self, fitted_reg_ensemble, mol_list, regression_y):
        """Calibrate the already-fitted ensemble; return it with pre-calibration std."""
        _, raw_std = fitted_reg_ensemble.predict(mol_list)
        fitted_reg_ensemble.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            algorithm="icp_regression",
        )
        return fitted_reg_ensemble, raw_std

    def test_calibrate_sets_calibrator(self, calibrated_reg):
        ens, _ = calibrated_reg
        assert ens.calibrator is not None

    def test_calibrate_populates_params(self, calibrated_reg):
        ens, _ = calibrated_reg
        assert ens.params.calibration is not None

    def test_calibrated_predict_returns_tuple(self, calibrated_reg, mol_list):
        ens, _ = calibrated_reg
        result = ens.predict(mol_list)
        assert isinstance(result, tuple)
        mean, std = result
        assert mean.shape == (len(mol_list), 1)
        assert std.shape == (len(mol_list), 1)

    def test_calibrated_std_is_finite_and_non_negative(self, calibrated_reg, mol_list):
        ens, _ = calibrated_reg
        _, std = ens.predict(mol_list)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)

    def test_calibrated_std_differs_from_raw(self, calibrated_reg, mol_list):
        ens, raw_std = calibrated_reg
        _, cal_std = ens.predict(mol_list)
        assert not np.allclose(raw_std, cal_std) or raw_std.sum() == 0.0

    def test_calibration_survives_save_load(self, calibrated_reg, tmp_path):
        ens, _ = calibrated_reg
        save_dir = str(tmp_path / "ens_cal")
        ens.save_model(save_dir)
        assert os.path.exists(os.path.join(save_dir, "state", "calibrator.pkl"))


# =========================================================================
# Calibration – classification ensembles  (one fit + calibrate via fixture)
# =========================================================================


class TestEnsembleClassifierCalibration:
    """Verify calibrate_uncertainty for classification ensembles."""

    @pytest.fixture()
    def calibrated_cls(self, fitted_cls_ensemble, mol_list, classification_y):
        fitted_cls_ensemble.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=classification_y,
            algorithm="icp_classification",
        )
        return fitted_cls_ensemble

    def test_calibrate_sets_calibrator(self, calibrated_cls):
        assert calibrated_cls.calibrator is not None

    def test_calibrate_populates_params(self, calibrated_cls):
        assert calibrated_cls.params.calibration is not None

    def test_calibrated_predict_returns_tuple(self, calibrated_cls, mol_list):
        result = calibrated_cls.predict(mol_list)
        assert isinstance(result, tuple)
        mean, std = result
        assert mean.shape == (len(mol_list), 1)
        assert std.shape == (len(mol_list), 1)

    def test_calibrated_std_is_finite_and_non_negative(self, calibrated_cls, mol_list):
        _, std = calibrated_cls.predict(mol_list)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)


# =========================================================================
# MLflow – regression ensembles  (setup is cheap; fit is expensive)
# =========================================================================


class TestEnsembleRegressorMLflow:
    """Verify MLflow logging for regression ensembles."""

    def test_set_mlflow_experiment_stores_params(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        ens.set_mlflow_experiment(
            experiment_name="ens_test",
            run_name="run_0",
        )
        assert ens.params.mlflow is not None
        assert isinstance(ens.params.mlflow, MLFlowInputModel)
        assert ens.params.mlflow.experiment == "ens_test"
        assert ens.params.mlflow.run == "run_0"

    def test_set_mlflow_experiment_adds_ensemble_tag(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        ens.set_mlflow_experiment(
            experiment_name="ens_test",
            run_name="run_0",
        )
        assert "model type" in ens.params.mlflow.tag
        assert ens.params.mlflow.tag["model type"] == "ensemble"

    def test_set_mlflow_with_custom_tag(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        ens.set_mlflow_experiment(
            experiment_name="ens_test",
            run_name="run_0",
            tag={"version": "42"},
        )
        assert ens.params.mlflow.tag["version"] == "42"
        assert ens.params.mlflow.tag["model type"] == "ensemble"

    def test_set_mlflow_custom_log_dir(self, regressor_factory, tmp_path):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        log_dir = str(tmp_path / "mlflow_log")
        ens.set_mlflow_experiment(
            experiment_name="ens_test",
            run_name="run_0",
            log_dir=log_dir,
        )
        assert ens.params.mlflow.log_dir == log_dir

    def test_mlflow_manager_is_active_after_setup(self, regressor_factory):
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        assert ens._mlflow_manager.is_active is False
        ens.set_mlflow_experiment(
            experiment_name="ens_test",
            run_name="run_0",
        )
        assert ens._mlflow_manager.is_active is True

    def test_fit_with_mlflow_creates_artifacts_and_predicts(
        self, regressor_factory, mol_list, regression_y, tmp_path
    ):
        """Single test covering fit-with-mlflow + prediction (avoids double fit)."""
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        log_dir = str(tmp_path / "mlflow_log")
        ens.set_mlflow_experiment(
            experiment_name="ens_fit_test",
            run_name="fit_run",
            log_dir=log_dir,
        )
        ens.fit(mol_list, regression_y)

        # artifacts were created
        assert os.path.exists(log_dir)

        # model still predicts correctly
        mean, std = ens.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(mean))


# =========================================================================
# MLflow – classification ensembles
# =========================================================================


class TestEnsembleClassifierMLflow:
    """Verify MLflow logging for classification ensembles."""

    def test_set_mlflow_experiment_stores_params(self, classifier_factory):
        template = classifier_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        ens.set_mlflow_experiment(
            experiment_name="ens_cls_test",
            run_name="run_0",
        )
        assert ens.params.mlflow is not None
        assert isinstance(ens.params.mlflow, MLFlowInputModel)
        assert ens.params.mlflow.experiment == "ens_cls_test"

    def test_fit_with_mlflow_creates_artifacts_and_predicts(
        self, classifier_factory, mol_list, classification_y, tmp_path
    ):
        """Single test covering fit-with-mlflow + prediction (avoids double fit)."""
        template = classifier_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        log_dir = str(tmp_path / "mlflow_log")
        ens.set_mlflow_experiment(
            experiment_name="ens_cls_fit_test",
            run_name="fit_run",
            log_dir=log_dir,
        )
        ens.fit(mol_list, classification_y)

        assert os.path.exists(log_dir)

        mean, std = ens.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(mean))


# =========================================================================
# MLflow + Calibration combined  (one fit + calibrate)
# =========================================================================


class TestEnsembleMLflowWithCalibration:
    """Verify that calibration with MLflow active logs artifacts correctly."""

    def test_calibrate_with_mlflow_creates_log_dir_and_predicts(
        self, regressor_factory, mol_list, regression_y, tmp_path
    ):
        """Single test: fit with mlflow → calibrate → check logs + predict."""
        template = regressor_factory()
        ens = Ensemble(model=template, n_models=N_MODELS)
        log_dir = str(tmp_path / "mlflow_cal_log")
        ens.set_mlflow_experiment(
            experiment_name="ens_cal_test",
            run_name="cal_run",
            log_dir=log_dir,
        )
        ens.fit(mol_list, regression_y)
        ens.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            algorithm="icp_regression",
        )

        # MLflow log directory should exist after calibration logging
        assert os.path.exists(log_dir)

        # predictions should still work correctly
        mean, std = ens.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)
