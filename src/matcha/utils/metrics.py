"""Metrics computation for regression and classification model evaluation.

Provides functions for calculating standard performance metrics, handling
censored data, and computing enrichment factors for ranking tasks.
"""

from matcha.sklearn.base_sklearn_model import BaseScikitLearnModel
from matcha.utils.sanitize import ensure_1d_array
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
import numpy as np
from scipy.stats import spearmanr
import copy


def process_censor(
    labels: np.ndarray, predictions: np.ndarray, censor: list[str]
) -> np.ndarray:
    """Apply censoring constraints to predictions based on label bounds and censor indicators."""
    lt = [i for i, s in enumerate(censor) if "<" in s]
    gt = [i for i, s in enumerate(censor) if ">" in s]

    labels = ensure_1d_array(labels)
    predictions = ensure_1d_array(predictions)

    predictions_censor = copy.deepcopy(predictions)

    if len(lt) > 0:
        bool_mask = np.zeros((len(censor)), dtype=bool)
        bool_mask[lt] = True
        predictions_censor = np.where(
            (predictions < labels) & bool_mask, labels, predictions_censor
        )

    if len(gt) > 0:
        bool_mask = np.zeros(len(censor), dtype=bool)
        bool_mask[gt] = True
        predictions_censor = np.where(
            (predictions > labels) & bool_mask, labels, predictions_censor
        )

    return predictions_censor


def process_regression(
    labels: np.ndarray, predictions: np.ndarray, log10: bool = False
) -> np.ndarray:
    """Calculate regression metrics including R2, RMSE, MAE, Spearman correlation, and fold accuracy."""
    labels = labels.copy()
    predictions = predictions.copy()

    labels = ensure_1d_array(labels)
    predictions = ensure_1d_array(predictions)

    if labels.ndim > 1:
        labels = labels[:, 0]
    if predictions.ndim > 1:
        predictions = predictions[:, 0]

    # Remove rows where labels is NaN
    mask = ~np.isnan(labels)
    labels = labels[mask]
    predictions = predictions[mask]

    if log10:
        labels = np.log10(labels)
        predictions = np.log10(predictions)

    output = {}
    output["R2"] = round(r2_score(labels, predictions), 3)
    output["RMSE"] = round(np.sqrt(mean_squared_error(labels, predictions)), 3)
    output["MAE"] = round(mean_absolute_error(labels, predictions), 3)
    output["SPEARMAN_R"] = round(spearmanr(labels, predictions)[0], 3)
    residuals = abs(labels - predictions)

    if log10:
        output["Within2Fold"] = round(
            (sum(x <= np.log10(2) for x in residuals) * 100 / len(residuals)), 3
        )
        output["Within3Fold"] = round(
            (sum(x <= np.log10(3) for x in residuals) * 100 / len(residuals)), 3
        )
    else:
        value2 = abs(predictions / labels)
        count2 = sum((x > 1.0 / 2) and (x < 2) for x in value2)
        output["Within2Fold"] = round(count2 / len(labels) * 100, 3)
        value3 = abs(predictions / labels)
        count3 = sum((x > 1.0 / 3) and (x < 3) for x in value3)
        output["Within3Fold"] = round(count3 / len(labels) * 100, 3)

    return output


def enrichment_factor_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Function to compute Enrichment Factor using precomputed binary labels
    according to the threshold set in process_ranking
    """

    y_pred = y_prob.copy()
    y_pred[y_pred < np.percentile(y_prob, 90)] = 0
    y_pred[y_pred >= np.percentile(y_prob, 90)] = 1

    compounds_at_k = np.sum(y_pred)
    total_compounds = len(y_true)
    total_actives = np.sum(y_true)
    tp_at_k = len(np.where(y_true + y_pred == 2)[0])

    return (tp_at_k / compounds_at_k) * (total_actives / total_compounds)


def process_classification(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    model: BaseScikitLearnModel | None = None,
) -> np.ndarray:
    """Calculate classification metrics including accuracy, F1, precision, recall, ROC-AUC, and PR-AUC."""
    output = {}
    labels = labels.copy()
    predictions = predictions.copy()
    probabilities = probabilities.copy()

    labels = ensure_1d_array(labels)
    predictions = ensure_1d_array(predictions)
    probabilities = ensure_1d_array(probabilities)

    if model is not None:
        labels = model._datamodule._label_encoder._all_to_categorical(labels)

    if labels.ndim > 1:
        labels = labels[:, 0]
    if predictions.ndim > 1:
        predictions = predictions[:, 0]
    if probabilities.ndim > 1:
        probabilities = probabilities[:, 0]

    # Remove rows where labels is NaN
    mask = ~np.isnan(labels)
    labels = labels[mask]
    predictions = predictions[mask]
    probabilities = probabilities[mask]

    output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["matthews_corrcoef"] = matthews_corrcoef(labels, predictions)
    output["f1_score"] = f1_score(labels, predictions)
    output["cohen_kappa"] = cohen_kappa_score(labels, predictions)
    output["precision"] = precision_score(labels, predictions)
    output["recall"] = recall_score(labels, predictions)

    output["roc_auc"] = roc_auc_score(labels, probabilities)
    output["pr_auc"] = average_precision_score(labels, probabilities)
    output["ef_10"] = enrichment_factor_score(labels, probabilities)

    return output
