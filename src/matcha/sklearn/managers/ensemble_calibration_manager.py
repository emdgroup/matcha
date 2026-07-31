import numpy as np

from matcha.calibration.base_calibration import CalibrationRegistry
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.calibration import CalibratorModel


class EnsembleCalibrationManager:
    """Manages calibration for ensemble models.

    Unlike :class:`UncertaintyManager` (which handles MC-dropout-based
    uncertainty for single models), this manager works with the ensemble's
    built-in variance across members as the uncertainty signal.
    """

    def __init__(self):
        self._calibrator = None
        self.logger = get_default_logger("ENSEMBLE_CALIBRATION")

    @property
    def calibrator(self):
        """The calibrator object, if one has been fitted."""
        return self._calibrator

    @property
    def params(self) -> CalibratorModel | None:
        """The calibration params, or None if no calibrator exists."""
        if self._calibrator is not None:
            return self._calibrator.params
        return None

    def create_calibrator(
        self,
        name: str = "inductive_conformal",
        params: dict | None = None,
        config: CalibratorModel | None = None,
    ) -> None:
        """Factory method to create a calibrator instance.

        :param str name: name of the calibration algorithm
        :param dict | None params: parameters for the calibrator
        :param CalibratorModel | None config: optional config (overrides name/params)
        """
        if params is None:
            params = {}
        if config is not None:
            params = config.model_dump()
            name = params.pop("calibrator_type")

        self._calibrator = CalibrationRegistry[name](**params)
        self.logger.info(f"Created calibrator: {name}")

    def calibrate(
        self,
        ensemble,
        calibration_mols,
        calibration_y: np.ndarray,
        algorithm: str = "inductive_conformal",
        algorithm_args: dict | None = None,
    ) -> None:
        """Calibrate uncertainty estimates using a calibration set.

        Performs ensemble prediction to get mean and std, creates a
        calibrator, and fits it on the calibration data.

        :param ensemble: the Ensemble instance
        :param calibration_mols: molecules for calibration
        :param np.ndarray calibration_y: true labels for calibration
        :param str algorithm: calibration algorithm name
        :param dict | None algorithm_args: arguments for the calibration algorithm
        """
        if algorithm_args is None:
            algorithm_args = {"confidence_alpha": 0.2}

        self.logger.info("Calibration: beginning process")
        preds, std = ensemble.predict(calibration_mols)
        self.create_calibrator(algorithm, algorithm_args)

        if ensemble.is_classifier and ensemble.model_box[0].has_class_labels():
            calibration_y = ensemble.encode_y(calibration_y)

        self._calibrator.fit(calibration_y, preds, std)

        # Sync calibration params to the ensemble
        ensemble.params.calibration = self._calibrator.params

        # Log to MLflow if active
        if ensemble._mlflow_manager.is_active:
            ensemble._mlflow_manager.log_calibrator(self._calibrator)

        self.logger.info("Calibration: uncertainty calibrated")

    def compute_uncertainty(self, std: np.ndarray) -> np.ndarray:
        """Adjust raw ensemble std using the fitted calibrator.

        :param np.ndarray std: raw standard deviation from ensemble predictions
        :return np.ndarray: calibrated uncertainty estimates
        """
        if self._calibrator is not None:
            return self._calibrator.compute_uncertainty(std)
        return std
