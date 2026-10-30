from abc import abstractmethod
from matcha.calibration.base_calibration import BaseCalibration, CalibrationRegistry
import numpy as np
from matcha.utils.schemas import (
    ICPRegressionInputModel,
    ICPClassificationInputModel,
)


class ICPCalibration(BaseCalibration):
    """
    Abstract base class for Inductive Conformal Prediction (ICP) calibration.

    This class provides the general framework for ICP, including methods to fit
    the calibration model using nonconformity scores and compute calibrated
    uncertainty estimates. Subclasses must implement the specific nonconformity
    score computation for regression or classification tasks.

    Supports multitask learning where each task is calibrated independently.
    NaN values in y_true are masked out per-task during calibration.

    The quantile calculation includes finite-sample correction to guarantee
    valid coverage: q = ceil((n+1)(1-α)) / n
    """

    @abstractmethod
    def _compute_nonconformity_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_error: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes the nonconformity scores for calibration.

        :param np.ndarray y_true: The true target values.
        :param np.ndarray y_pred: The predicted target values.
        :param np.ndarray | None y_error: Optional errors associated with predictions.

        :return: An array of nonconformity scores.
        """
        pass

    def _compute_quantile_single(
        self,
        scores: np.ndarray,
    ) -> float:
        """
        Computes the quantile for a single task's nonconformity scores with finite-sample correction.

        :param np.ndarray scores: The nonconformity scores for a single task.

        :return: The computed quantile value.
        """
        # Mask out NaN values
        valid_mask = ~np.isnan(scores)
        valid_scores = scores[valid_mask]
        n = len(valid_scores)

        if n == 0:
            return np.nan

        # Finite-sample correction for coverage guarantee
        adjusted_level = min(
            np.ceil((n + 1) * (1 - self.params.confidence_alpha)) / n, 1.0
        )
        return np.nanquantile(valid_scores, adjusted_level, method="higher")

    def _compute_quantile(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_error: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes the quantile for each task's nonconformity scores with finite-sample correction.

        :param np.ndarray y_true: The true target values, shape (n_samples,) or (n_samples, n_tasks).
        :param np.ndarray y_pred: The predicted target values.
        :param np.ndarray | None y_error: Optional errors associated with predictions.

        :return: Array of quantile values, one per task.
        """
        scores = self._compute_nonconformity_scores(y_true, y_pred, y_error)

        # Ensure 2D
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        n_tasks = scores.shape[1]
        quantiles = np.zeros(n_tasks)

        for task_idx in range(n_tasks):
            quantiles[task_idx] = self._compute_quantile_single(scores[:, task_idx])

        return quantiles

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_error: np.ndarray | None = None,
    ) -> None:
        """
        Fits the ICP calibration model using true and predicted values.

        :param np.ndarray y_true: The true target values, shape (n_samples,) or (n_samples, n_tasks).
        :param np.ndarray y_pred: The predicted target values.
        :param np.ndarray | None y_error: Optional un-calibrated uncertainties to scale.
        """
        self.params.quantile = self._compute_quantile(y_true, y_pred, y_error)
        self.is_fitted = True

    @abstractmethod
    def compute_uncertainty(self, input: np.ndarray) -> np.ndarray:
        """
        Computes calibrated uncertainty based on the input data.

        :param np.ndarray input: The input data for which to compute uncertainty.

        :return: An array of calibrated uncertainty estimates.
        """
        pass

    def _check_is_fitted(self) -> None:
        """Raises an error if the model has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before computing uncertainty. "
                "Call fit() first."
            )


@CalibrationRegistry.register("icp_regression")
class ICPRegressionCalibration(ICPCalibration):
    """
    Inductive Conformal Prediction calibration for regression tasks.

    Computes calibrated prediction intervals using normalized absolute residuals
    as nonconformity scores. Uses heteroscedastic calibration where residuals
    are normalized by model uncertainty estimates.

    Supports multitask regression where each task is calibrated independently.
    NaN values in y_true are masked out per-task during calibration.

    Example:
        >>> calibrator = ICPRegressionCalibration(confidence_alpha=0.1)
        >>> calibrator.fit(y_true_cal, y_pred_cal, y_error_cal)
        >>> uncertainty = calibrator.compute_uncertainty(y_error_test)
        >>> # Prediction interval: [y_pred - uncertainty, y_pred + uncertainty]
    """

    def __init__(self, confidence_alpha: float = 0.2):
        """
        Initialize ICP Regression calibrator.

        :param float confidence_alpha: The significance level (1 - coverage).
            E.g., alpha=0.1 targets 90% coverage. Defaults to 0.2 (80% coverage).
        """
        super().__init__()
        self.params = ICPRegressionInputModel(
            confidence_alpha=confidence_alpha,
            quantile=0.0,
        )

    def _compute_nonconformity_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_error: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes normalized absolute residuals as nonconformity scores.

        Residuals are normalized by the uncertainty estimates (heteroscedastic mode).
        NaN values in y_true propagate to NaN scores which are handled during quantile computation.

        :param np.ndarray y_true: The true target values, shape (n_samples,) or (n_samples, n_tasks).
        :param np.ndarray y_pred: The predicted target values.
        :param np.ndarray y_error: Model uncertainty estimates for normalization.

        :return: An array of nonconformity scores, same shape as y_true.
        """
        residuals = np.abs(y_true - y_pred)

        if y_error is not None:
            # Normalized residuals for heteroscedastic calibration
            residuals = residuals / (y_error + 1e-6)

        return residuals

    def compute_uncertainty(self, input: np.ndarray) -> np.ndarray:
        """
        Computes calibrated uncertainty estimates.

        Scales input uncertainties by the calibration quantile (per-task).

        :param np.ndarray input: Uncalibrated uncertainty estimates,
            shape (n_samples,) or (n_samples, n_tasks).

        :return: An array of calibrated uncertainty estimates.
        """
        self._check_is_fitted()

        # Ensure input is 2D for broadcasting
        input_2d = input.reshape(-1, 1) if input.ndim == 1 else input

        # Broadcast multiply: (n_samples, n_tasks) * (n_tasks,)
        calibrated = input_2d * self.params.quantile

        # Return same shape as input
        return calibrated.squeeze() if input.ndim == 1 else calibrated


@CalibrationRegistry.register("icp_classification")
class ICPClassificationCalibration(ICPCalibration):
    """
    Inductive Conformal Prediction calibration for multitask binary classification.

    Each task is treated as an independent binary classification problem
    (sigmoid activation, not softmax). Computes calibrated prediction sets
    using 1 - probability of true class as nonconformity scores.

    Supports multitask classification where each task is calibrated independently.
    NaN values in y_true are masked out per-task during calibration.

    Example:
        >>> calibrator = ICPClassificationCalibration(confidence_alpha=0.1)
        >>> calibrator.fit(y_true_cal, y_pred_proba_cal)  # y_pred_proba: P(class=1) for each task
        >>> p_values = calibrator.compute_uncertainty(y_pred_proba_test)
        >>> # Prediction set for task i: include class 1 if p_values[:, i] > alpha
    """

    def __init__(self, confidence_alpha: float = 0.2):
        """
        Initialize ICP Classification calibrator for multitask binary classification.

        :param float confidence_alpha: The significance level for prediction sets.
            E.g., alpha=0.1 targets 90% coverage. Defaults to 0.2.
        """
        super().__init__()
        self.params = ICPClassificationInputModel(
            confidence_alpha=confidence_alpha,
            quantile=0.0,
        )
        # Store calibration scores per task: list of arrays
        self._calibration_scores: list[np.ndarray] | None = None

    def _compute_nonconformity_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_error: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes nonconformity scores for multitask binary classification.

        For each task, uses 1 - P(true class) as the nonconformity score.
        NaN values in y_true propagate to NaN scores.

        :param np.ndarray y_true: True binary labels, shape (n_samples,) or (n_samples, n_tasks).
            Values should be 0 or 1 (or NaN for missing).
        :param np.ndarray y_pred: Predicted probabilities for class 1,
            shape (n_samples,) or (n_samples, n_tasks).
        :param np.ndarray | None y_error: Unused for classification.

        :return: An array of nonconformity scores, same shape as y_true.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Ensure 2D
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # For binary classification: P(true class) = y_pred if y_true=1, else 1-y_pred
        # Nonconformity score = 1 - P(true class)
        prob_true_class = np.where(y_true == 1, y_pred, 1 - y_pred)
        scores = 1 - prob_true_class

        # Propagate NaN from y_true
        scores = np.where(np.isnan(y_true), np.nan, scores)

        return scores

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_error: np.ndarray | None = None,
    ) -> None:
        """
        Fits the ICP classification calibration model for multitask binary classification.

        Stores the calibration nonconformity scores per task for computing p-values.

        :param np.ndarray y_true: True binary labels, shape (n_samples,) or (n_samples, n_tasks).
        :param np.ndarray y_pred: Predicted probabilities for class 1.
        :param np.ndarray | None y_error: Unused for classification.
        """
        scores = self._compute_nonconformity_scores(y_true, y_pred, y_error)

        # Ensure 2D
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        # Store valid (non-NaN) calibration scores per task
        n_tasks = scores.shape[1]
        self._calibration_scores = []
        for task_idx in range(n_tasks):
            task_scores = scores[:, task_idx]
            valid_mask = ~np.isnan(task_scores)
            self._calibration_scores.append(task_scores[valid_mask])

        self.params.quantile = self._compute_quantile(y_true, y_pred, y_error)
        self.is_fitted = True

    def compute_uncertainty(self, input: np.ndarray) -> np.ndarray:
        """
        Computes p-values for each task based on predicted probabilities.

        For each sample and each task, computes p-values for both class 0 and class 1.
        The p-value is the proportion of calibration scores >= test nonconformity score.

        :param np.ndarray input: Predicted probabilities for class 1,
            shape (n_samples,) or (n_samples, n_tasks).

        :return: Array of p-values for class 1, shape (n_samples, n_tasks).
            To get p-values for class 0, use the complementary probability.
        """
        self._check_is_fitted()

        input = np.asarray(input)
        if input.ndim == 1:
            input = input.reshape(-1, 1)

        n_samples, n_tasks = input.shape

        if len(self._calibration_scores) != n_tasks:
            raise ValueError(
                f"Number of tasks in input ({n_tasks}) does not match "
                f"number of tasks during fitting ({len(self._calibration_scores)})"
            )

        # Compute p-values for class 1 for each task
        p_values = np.zeros((n_samples, n_tasks))

        for task_idx in range(n_tasks):
            cal_scores = self._calibration_scores[task_idx]
            n_cal = len(cal_scores)

            if n_cal == 0:
                # No valid calibration data for this task
                p_values[:, task_idx] = np.nan
                continue

            # Nonconformity score if class 1 is the true class
            test_scores = 1 - input[:, task_idx]

            # P-value: (# calibration scores >= test score + 1) / (n_cal + 1)
            for i, score in enumerate(test_scores):
                p_values[i, task_idx] = (np.sum(cal_scores >= score) + 1) / (n_cal + 1)

        return p_values

    def compute_p_values_both_classes(
        self, input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes p-values for both classes (0 and 1) for each task.

        :param np.ndarray input: Predicted probabilities for class 1,
            shape (n_samples,) or (n_samples, n_tasks).

        :return: Tuple of (p_values_class_0, p_values_class_1), each shape (n_samples, n_tasks).
        """
        self._check_is_fitted()

        input = np.asarray(input)
        if input.ndim == 1:
            input = input.reshape(-1, 1)

        n_samples, n_tasks = input.shape

        p_values_0 = np.zeros((n_samples, n_tasks))
        p_values_1 = np.zeros((n_samples, n_tasks))

        for task_idx in range(n_tasks):
            cal_scores = self._calibration_scores[task_idx]
            n_cal = len(cal_scores)

            if n_cal == 0:
                p_values_0[:, task_idx] = np.nan
                p_values_1[:, task_idx] = np.nan
                continue

            for i in range(n_samples):
                # P-value for class 1: nonconformity score = 1 - P(class=1)
                score_1 = 1 - input[i, task_idx]
                p_values_1[i, task_idx] = (np.sum(cal_scores >= score_1) + 1) / (
                    n_cal + 1
                )

                # P-value for class 0: nonconformity score = 1 - P(class=0) = 1 - (1 - input) = input
                score_0 = input[i, task_idx]
                p_values_0[i, task_idx] = (np.sum(cal_scores >= score_0) + 1) / (
                    n_cal + 1
                )

        return p_values_0, p_values_1

    def predict_sets(self, input: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """
        Generates prediction confidence indicators at the specified significance level.

        For each sample and each task, returns an integer indicating the
        prediction status:

        *  ``1``  — **confident**: exactly one class is in the prediction set.
        *  ``0``  — **uncertain**: both classes are in the prediction set.
        * ``-1``  — **unusual**: neither class is in the prediction set
          (signals possible distribution shift).

        :param np.ndarray input: Predicted probabilities for class 1,
            shape (n_samples,) or (n_samples, n_tasks).
        :param float | None alpha: Significance level. If None, uses confidence_alpha.

        :return: Integer array of shape (n_samples, n_tasks) with values in {-1, 0, 1}.
        """
        self._check_is_fitted()

        if alpha is None:
            alpha = self.params.confidence_alpha

        p_values_0, p_values_1 = self.compute_p_values_both_classes(input)

        included_0 = p_values_0 > alpha  # class 0 in set
        included_1 = p_values_1 > alpha  # class 1 in set
        set_size = included_0.astype(int) + included_1.astype(int)

        # size 1 → confident (1), size 2 → uncertain (0), size 0 → unusual (-1)
        result = np.where(set_size == 1, 1, np.where(set_size == 2, 0, -1))

        return result
