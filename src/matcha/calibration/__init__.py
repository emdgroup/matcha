"""Calibration methods for uncertainty quantification in predictive models.

This module provides post-hoc calibration techniques that transform raw model
predictions into calibrated uncertainty estimates. Two main approaches are supported:

- **Inductive Conformal Prediction (ICP):** Produces prediction intervals (regression)
  or prediction sets (classification) with finite-sample coverage guarantees.
- **Error Models (EM):** Train secondary classifiers to predict whether a primary
  model's predictions fall within an acceptable error range.

All calibration methods follow the fit/predict pattern defined by
:class:`BaseCalibration` and are registered via :data:`CalibrationRegistry`.
"""

from matcha.calibration.base_calibration import BaseCalibration
from matcha.calibration.inductive_conformal import (
    ICPRegressionCalibration,
    ICPClassificationCalibration,
)
from matcha.calibration.error_model import (
    EMCalibration,
    EMRegressionCalibration,
    EMClassificationCalibration,
)

__all__ = [
    "BaseCalibration",
    "ICPRegressionCalibration",
    "ICPClassificationCalibration",
    "EMCalibration",
    "EMRegressionCalibration",
    "EMClassificationCalibration",
]
