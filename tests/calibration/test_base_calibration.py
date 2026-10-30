"""Tests for BaseCalibration and CalibrationRegistry."""

import os
import numpy as np
import pytest

from matcha.calibration.base_calibration import BaseCalibration, CalibrationRegistry
from matcha.calibration.inductive_conformal import (
    ICPRegressionCalibration,
    ICPClassificationCalibration,
)
from matcha.calibration.error_model import (
    EMRegressionCalibration,
    EMClassificationCalibration,
)


# ===================================================================
# BaseCalibration – abstract contract
# ===================================================================


class TestBaseCalibrationIsAbstract:
    """Verify that BaseCalibration cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseCalibration()


# ===================================================================
# is_fitted property
# ===================================================================


class TestIsFittedProperty:
    """Tests for the is_fitted property on concrete subclasses."""

    def test_initial_is_fitted_false(self):
        cal = ICPRegressionCalibration()
        assert cal.is_fitted is False

    def test_set_is_fitted_true(self):
        cal = ICPRegressionCalibration()
        cal.is_fitted = True
        assert cal.is_fitted is True

    def test_set_is_fitted_back_to_false(self):
        cal = ICPRegressionCalibration()
        cal.is_fitted = True
        cal.is_fitted = False
        assert cal.is_fitted is False


# ===================================================================
# save / load round-trip
# ===================================================================


class TestSaveAndLoad:
    """Tests for save_calibrator and from_folder round-trip."""

    def test_save_creates_file(self, tmp_path, regression_calibration_data):
        y_true, y_pred, y_error = regression_calibration_data
        cal = ICPRegressionCalibration(confidence_alpha=0.1)
        cal.fit(y_true, y_pred, y_error)

        save_dir = str(tmp_path / "cal_save")
        cal.save_calibrator(save_dir)

        assert os.path.isfile(os.path.join(save_dir, "calibrator.pkl"))

    def test_load_restores_params(self, tmp_path, regression_calibration_data):
        y_true, y_pred, y_error = regression_calibration_data
        cal = ICPRegressionCalibration(confidence_alpha=0.15)
        cal.fit(y_true, y_pred, y_error)

        save_dir = str(tmp_path / "cal_save2")
        cal.save_calibrator(save_dir)

        loaded = ICPRegressionCalibration.from_folder(save_dir)
        assert loaded.params.confidence_alpha == pytest.approx(0.15)
        np.testing.assert_array_almost_equal(
            loaded.params.quantile, cal.params.quantile
        )

    def test_load_classification_params(
        self, tmp_path, classification_calibration_data
    ):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration(confidence_alpha=0.05)
        cal.fit(y_true, y_pred)

        save_dir = str(tmp_path / "cal_cls")
        cal.save_calibrator(save_dir)

        loaded = ICPClassificationCalibration.from_folder(save_dir)
        assert loaded.params.confidence_alpha == pytest.approx(0.05)


# ===================================================================
# CalibrationRegistry
# ===================================================================


class TestCalibrationRegistry:
    """Verify classes are registered correctly."""

    def test_icp_regression_registered(self):
        assert "icp_regression" in CalibrationRegistry

    def test_icp_classification_registered(self):
        assert "icp_classification" in CalibrationRegistry

    def test_error_model_regression_registered(self):
        assert "error_model_regression" in CalibrationRegistry

    def test_error_model_classification_registered(self):
        assert "error_model_classification" in CalibrationRegistry

    def test_registry_returns_correct_class_icp_reg(self):
        assert CalibrationRegistry["icp_regression"] is ICPRegressionCalibration

    def test_registry_returns_correct_class_icp_cls(self):
        assert CalibrationRegistry["icp_classification"] is ICPClassificationCalibration

    def test_registry_returns_correct_class_em_reg(self):
        assert CalibrationRegistry["error_model_regression"] is EMRegressionCalibration

    def test_registry_returns_correct_class_em_cls(self):
        assert (
            CalibrationRegistry["error_model_classification"]
            is EMClassificationCalibration
        )

    def test_registry_raises_for_unknown(self):
        with pytest.raises(ValueError, match="is not valid"):
            CalibrationRegistry["nonexistent_calibrator"]
