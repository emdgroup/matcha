import numpy as np

from matcha.calibration.base_calibration import CalibrationRegistry
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.calibration import CalibratorModel


class UncertaintyManager:
    """Manages MC Dropout uncertainty estimation and calibration."""

    def __init__(self):
        self._calibrator = None
        self.logger = get_default_logger("UNCERTAINTY")

    @property
    def calibrator(self):
        """The calibrator object, if one has been fitted."""
        return self._calibrator

    @property
    def params(self) -> CalibratorModel | None:
        """The calibration params, or None if no calibrator has been fitted."""
        if self._calibrator is not None:
            return self._calibrator.params
        return None

    def create_calibrator(
        self, name: str, params: dict, config: CalibratorModel | None = None
    ):
        """Factory method to create a calibrator instance.

        :param str name: name of the calibrator algorithm
        :param dict params: parameters for the calibrator
        :param CalibratorModel | None config: optional config object (overrides name/params)
        """
        if config is not None:
            params = config.model_dump()
            name = params["calibrator_type"]
            params.pop("calibrator_type")

        self._calibrator = CalibrationRegistry[name](**params)

    def compute(
        self,
        model_instance,
        x,
        num_iterations: int = 10,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Computes uncertainty via Monte Carlo dropout for a test set.

        :param model_instance: the sklearn model instance
        :param x: input to compute uncertainty for
        :param int num_iterations: how many iterations of dropout to do
        :param str | None accelerator: hardware to use for predictions
        :param int | None devices: how many resources to use
        :param int | None batch_size: batch size to use
        :return np.ndarray: array with the uncertainty for each prediction
        """
        self.logger.info("Uncertainty estimation: beginning process")

        model_instance._model.switch_mc_dropout()
        pred_box = []
        for _ in range(num_iterations):
            pred_box.append(
                model_instance._default_predict(x, accelerator, devices, batch_size)
            )
        model_instance._model.switch_mc_dropout()
        pred_box = np.stack(pred_box, axis=2)

        std = np.std(pred_box, axis=2)

        if self._calibrator is not None:
            self.logger.info("Adjusting uncertainty estimates with the calibrator")
            std = self._calibrator.compute_uncertainty(std)

        self.logger.info("Uncertainty estimation: finished")
        return std

    def calibrate(
        self,
        model_instance,
        calibration_mols,
        calibration_y: np.ndarray,
        num_iterations: int = 10,
        algorithm: str = "inductive_conformal",
        algorithm_args: dict | None = None,
    ):
        """Calibrate uncertainty estimates using a calibration set.

        :param model_instance: the sklearn model instance
        :param calibration_mols: molecules for calibration
        :param np.ndarray calibration_y: true labels for calibration
        :param int num_iterations: MC dropout iterations
        :param str algorithm: calibration algorithm name
        :param dict | None algorithm_args: arguments for the calibration algorithm
        """
        if algorithm_args is None:
            algorithm_args = {"confidence_alpha": 0.2}
        self.logger.info("Calibration: beginning process")
        preds = model_instance._default_predict(calibration_mols)
        std = self.compute(model_instance, calibration_mols, num_iterations)
        self.create_calibrator(algorithm, algorithm_args)

        if (
            model_instance.params.datamodule.is_classification
            and model_instance.datamodule._label_encoder.is_set()
        ):
            calibration_y = model_instance.encode_y(calibration_y)

        self._calibrator.fit(calibration_y, preds, std)

        # Log to MLflow if active
        if model_instance._mlflow_manager.is_active:
            model_instance._mlflow_manager.log_calibrator(self._calibrator)

        self.logger.info("Calibration: uncertainty calibrated")
