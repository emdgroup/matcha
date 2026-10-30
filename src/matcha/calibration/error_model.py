from abc import abstractmethod
from matcha.calibration.base_calibration import BaseCalibration, CalibrationRegistry
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.rdchem import Mol
from matcha.datamodules.classic.rdkit_engine import Engine
from matcha.utils.schemas import (
    EMRegressionInputModel,
    EMClassificationInputModel,
)


EM_REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "LogisticRegression": LogisticRegression,
}


class EMCalibration(BaseCalibration):
    """
    Abstract base class for Error Model calibration.

    This class provides the general framework for error model calibration,
    which trains a classifier to predict whether predictions are within
    an acceptable error threshold. Subclasses implement the specific error
    computation for regression or classification tasks.

    Supports multitask learning where each task is calibrated independently.
    NaN values in y_true are masked out per-task during calibration.
    """

    def __init__(self):
        super().__init__()
        self.engine = Engine(n_jobs=1)
        self.scaler_box: list[StandardScaler] = []
        self.error_ratio: list[float] = []
        self.n_compounds: list[int] = []
        self.model_box: list = []

    @abstractmethod
    def _compute_error_mask(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Computes a boolean mask indicating which predictions are within acceptable error.

        :param np.ndarray y_true: The true target values.
        :param np.ndarray y_pred: The predicted target values.

        :return: Boolean array where True = acceptable error, False = unacceptable error.
        """
        pass

    def _feature_engineer(
        self,
        mols: list[Mol],
        preds: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        """
        Engineers features for the error model from predictions, uncertainties, and molecules.

        :param list[Mol] mols: List of RDKit molecule objects.
        :param np.ndarray preds: Predicted values, shape (n_samples, n_tasks).
        :param np.ndarray std: Uncertainty estimates, shape (n_samples, n_tasks).

        :return: Feature matrix, shape (n_samples, n_features).
        """
        out = np.concatenate((preds, std), axis=1)

        if self.params.use_interaction:
            interaction = (preds[:, :, None] * std[:, None, :]).reshape(
                preds.shape[0], -1
            )
            out = np.concatenate((out, interaction), axis=1)

        if self.params.use_ecfp:
            n_jobs = len(mols) // 10000 if len(mols) > 10000 else 1
            ecfp = self.engine.get_ECFP(mols, n_jobs=n_jobs)
            out = np.concatenate((out, ecfp), axis=1)

        return out

    def fit(
        self,
        mols: list[Mol],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_error: np.ndarray,
    ) -> None:
        """
        Fits the error model calibration using molecules, true values, predictions, and uncertainties.

        :param list[Mol] mols: List of RDKit molecule objects.
        :param np.ndarray y_true: The true target values, shape (n_samples,) or (n_samples, n_tasks).
        :param np.ndarray y_pred: The predicted target values.
        :param np.ndarray y_error: Uncertainty estimates for predictions.
        """
        self.model_box = []
        self.error_ratio = []
        self.n_compounds = []
        self.scaler_box = []

        # Ensure 2D
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_error.ndim == 1:
            y_error = y_error.reshape(-1, 1)

        n_tasks = y_true.shape[1]

        for i in range(n_tasks):
            mask = ~np.isnan(y_true[:, i])
            y_true_i = y_true[mask]
            y_pred_i = y_pred[mask]
            y_error_i = y_error[mask]
            mols_i = [mols[x] for x in np.where(mask)[0]]

            x_i = self._feature_engineer(mols_i, y_pred_i, y_error_i)
            scaler = StandardScaler()
            x_i = scaler.fit_transform(x_i)
            self.scaler_box.append(scaler)

            error_mask = self._compute_error_mask(y_true_i[:, i], y_pred_i[:, i])

            if (
                np.mean(error_mask) != 1.0
                and np.mean(error_mask) != 0.0
                and len(y_true_i) >= self.params.min_compounds
            ):
                try:
                    model = EM_REGISTRY[self.params.algorithm](
                        random_state=0, **self.params.algorithm_params
                    )
                except Exception:
                    model = EM_REGISTRY[self.params.algorithm](
                        **self.params.algorithm_params
                    )
                model.fit(x_i, error_mask)
            else:
                model = None

            self.model_box.append(model)
            self.error_ratio.append(np.mean(error_mask))
            self.n_compounds.append(len(y_true_i))

        self.is_fitted = True

    def compute_uncertainty(
        self,
        mols: list[Mol],
        y_pred: np.ndarray,
        y_error: np.ndarray,
    ) -> np.ndarray:
        """
        Computes error model predictions (probability of being within acceptable error).

        :param list[Mol] mols: List of RDKit molecule objects.
        :param np.ndarray y_pred: Predicted values, shape (n_samples,) or (n_samples, n_tasks).
        :param np.ndarray y_error: Uncertainty estimates, shape (n_samples,) or (n_samples, n_tasks).

        :return: Array of probabilities, shape (n_samples, n_tasks).
            Values indicate probability of prediction being within acceptable error.
            Special values: -1 = model not trainable (all same class), -2 = insufficient data.
        """
        self._check_is_fitted()

        # Ensure 2D
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_error.ndim == 1:
            y_error = y_error.reshape(-1, 1)

        x = self._feature_engineer(mols, y_pred, y_error)
        labels_box = []

        for i, model in enumerate(self.model_box):
            if self.n_compounds[i] <= self.params.min_compounds:
                labels_box.append(np.zeros(x.shape[0]) - 2)
            elif model is None:
                labels_box.append(np.zeros(x.shape[0]) - 1)
            else:
                x_scaled = self.scaler_box[i].transform(x)
                labels_box.append(model.predict_proba(x_scaled)[:, 1])

        output = np.stack(labels_box, axis=1)

        return output

    def _check_is_fitted(self) -> None:
        """Raises an error if the model has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before computing uncertainty. "
                "Call fit() first."
            )


@CalibrationRegistry.register("error_model_regression")
class EMRegressionCalibration(EMCalibration):
    """
    Error Model calibration for regression tasks.

    Trains a classifier to predict whether predictions are within N-fold
    change of the true values. This is useful for error detection in
    regression tasks where fold-change is a meaningful measure.

    Supports multitask regression where each task is calibrated independently.
    NaN values in y_true are masked out per-task during calibration.

    Example:
        >>> calibrator = EMRegressionCalibration(fold=2.0)
        >>> calibrator.fit(mols, y_true_cal, y_pred_cal, y_error_cal)
        >>> proba = calibrator.compute_uncertainty(mols_test, y_pred_test, y_error_test)
        >>> # proba indicates probability of prediction being within 2-fold error
    """

    def __init__(
        self,
        algorithm: str = "GradientBoostingClassifier",
        algorithm_params: dict = None,
        use_ecfp: bool = True,
        use_interaction: bool = True,
        min_compounds: int = 50,
        fold: float = 2.0,
        log10: bool = False,
    ):
        """
        Initialize Error Model Regression calibrator.

        :param str algorithm: Classifier algorithm to use. Options: 'RandomForestClassifier',
            'GradientBoostingClassifier', 'KNeighborsClassifier', 'LogisticRegression'.
            Defaults to 'GradientBoostingClassifier'.
        :param dict algorithm_params: Additional parameters to pass to the classifier.
        :param bool use_ecfp: Whether to include ECFP fingerprints as features. Defaults to True.
        :param bool use_interaction: Whether to include interaction features between
            predictions and uncertainties. Defaults to True.
        :param int min_compounds: Minimum number of compounds required per task to train
            the error model. Defaults to 50.
        :param float fold: The fold-change threshold. E.g., fold=2 means within 2-fold.
            Defaults to 2.0.
        :param bool log10: If True, assumes labels and predictions are in log10 space.
            Defaults to False.
        """
        super().__init__()
        if algorithm_params is None:
            algorithm_params = {}
        self.params = EMRegressionInputModel(
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            use_ecfp=use_ecfp,
            use_interaction=use_interaction,
            min_compounds=min_compounds,
            fold=fold,
            log10=log10,
        )

    def _compute_error_mask(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Computes a boolean mask indicating which predictions are within N-fold change.

        :param np.ndarray y_true: The true target values.
        :param np.ndarray y_pred: The predicted target values.

        :return: Boolean array where True = within fold change, False = outside fold change.
        """
        y_true = np.asarray(y_true).copy()
        y_pred = np.asarray(y_pred).copy()

        # Flatten if needed
        if y_true.ndim > 1:
            y_true = y_true[:, 0]
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 0]

        fold = self.params.fold

        if self.params.log10:
            # In log10 space, fold change is measured by absolute difference
            residuals = np.abs(y_true - y_pred)
            within_mask = residuals <= np.log10(fold)
        else:
            # In linear space, fold change is measured by ratio
            ratio = np.abs(y_pred / y_true)
            within_mask = (ratio >= 1.0 / fold) & (ratio <= fold)

        return within_mask


@CalibrationRegistry.register("error_model_classification")
class EMClassificationCalibration(EMCalibration):
    """
    Error Model calibration for classification tasks.

    Trains a classifier to predict whether predicted probabilities are within
    an acceptable absolute gap from the true binary labels. This is useful
    for error detection in classification tasks.

    Supports multitask classification where each task is calibrated independently.
    NaN values in y_true are masked out per-task during calibration.

    Example:
        >>> calibrator = EMClassificationCalibration(error_threshold=0.5)
        >>> calibrator.fit(mols, y_true_cal, y_pred_proba_cal, y_error_cal)
        >>> proba = calibrator.compute_uncertainty(mols_test, y_pred_proba_test, y_error_test)
        >>> # proba indicates probability of prediction being within 0.5 of true label
    """

    def __init__(
        self,
        algorithm: str = "GradientBoostingClassifier",
        algorithm_params: dict = None,
        use_ecfp: bool = True,
        use_interaction: bool = True,
        min_compounds: int = 50,
        error_threshold: float = 0.5,
    ):
        """
        Initialize Error Model Classification calibrator.

        :param str algorithm: Classifier algorithm to use. Options: 'RandomForestClassifier',
            'GradientBoostingClassifier', 'KNeighborsClassifier', 'LogisticRegression'.
            Defaults to 'GradientBoostingClassifier'.
        :param dict algorithm_params: Additional parameters to pass to the classifier.
        :param bool use_ecfp: Whether to include ECFP fingerprints as features. Defaults to True.
        :param bool use_interaction: Whether to include interaction features between
            predictions and uncertainties. Defaults to True.
        :param int min_compounds: Minimum number of compounds required per task to train
            the error model. Defaults to 50.
        :param float error_threshold: The absolute gap threshold. Predictions are considered
            acceptable if |y_pred_proba - y_true| <= error_threshold.
            Defaults to 0.5.
        """
        super().__init__()
        if algorithm_params is None:
            algorithm_params = {}
        self.params = EMClassificationInputModel(
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            use_ecfp=use_ecfp,
            use_interaction=use_interaction,
            min_compounds=min_compounds,
            error_threshold=error_threshold,
        )

    def _compute_error_mask(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Computes a boolean mask indicating which predictions are within acceptable error.

        For classification, checks whether the absolute gap between predicted probability
        and true label is within the error threshold.

        :param np.ndarray y_true: The true binary labels (0 or 1).
        :param np.ndarray y_pred: The predicted probabilities for class 1.

        :return: Boolean array where True = within threshold, False = outside threshold.
        """
        y_true = np.asarray(y_true).copy()
        y_pred = np.asarray(y_pred).copy()

        # Flatten if needed
        if y_true.ndim > 1:
            y_true = y_true[:, 0]
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 0]

        # Absolute gap between predicted probability and true label
        gap = np.abs(y_pred - y_true)
        within_mask = gap <= self.params.error_threshold

        return within_mask
