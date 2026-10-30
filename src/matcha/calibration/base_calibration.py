"""Base class and registry for calibration methods."""

import numpy as np
from abc import abstractmethod, ABC
from matcha.utils.serialization import save_pickle, load_pickle
from matcha.utils.registry import ClassRegistry
import os


class BaseCalibration(ABC):
    """Abstract base class for calibration methods in predictive modeling.

    This class provides a framework for implementing calibration techniques,
    including methods for fitting the model, computing uncertainty, and
    saving/loading calibration parameters.

    Subclasses must implement :meth:`fit` and :meth:`compute_uncertainty`.
    Calibration parameters are stored in ``self.params`` and can be persisted
    via :meth:`save_calibrator` / :meth:`from_folder`.
    """

    def __init__(self):
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the calibration model has been fitted.

        :returns: True if :meth:`fit` has been called successfully.
        """
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool):
        self._is_fitted = value

    @abstractmethod
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit the calibration model to true and predicted values.

        :param np.ndarray y_true: Ground-truth target values.
        :param np.ndarray y_pred: Model predictions to calibrate against.
        """
        pass

    @abstractmethod
    def compute_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Compute calibrated uncertainty estimates.

        :param np.ndarray predictions: Raw model predictions or uncertainties.

        :returns: Calibrated uncertainty estimates.
        """
        pass

    def save_calibrator(self, path: str):
        """Save calibration parameters to disk.

        Creates the target directory if it does not exist and serializes
        ``self.params`` as a pickle file.

        :param str path: Directory path where ``calibrator.pkl`` will be saved.
        """
        os.makedirs(path, exist_ok=True)
        save_pickle(f"{path}/calibrator.pkl", self.params)

    @classmethod
    def from_folder(cls, path: str):
        """Load a calibrator from a previously saved folder.

        :param str path: Directory containing ``calibrator.pkl``.

        :returns: An instance of the calibration class with loaded parameters.
        """
        params = load_pickle(f"{path}/calibrator.pkl")
        out = cls()
        out.params = params
        return out


CalibrationRegistry = ClassRegistry()
