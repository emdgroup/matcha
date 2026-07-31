"""Tests for matcha.utils.metrics – process_censor, process_regression,
process_classification, and enrichment_factor_score.

Focuses on the *matcha-specific* aggregation logic (censoring, fold-accuracy,
enrichment factor) rather than re-testing sklearn metric implementations.
"""

import numpy as np

from matcha.utils.metrics import (
    process_censor,
    process_classification,
    process_regression,
)


# =========================================================================
# process_censor
# =========================================================================


class TestProcessCensor:
    """Tests for censoring logic that clips predictions at label bounds."""

    def test_less_than_censor_clips_upward(self):
        """If censor is '<', predictions below the label are clipped *up* to the label."""
        labels = np.array([5.0, 5.0, 5.0])
        predictions = np.array([3.0, 6.0, 5.0])
        censor = ["<", "<", "<"]
        result = process_censor(labels, predictions, censor)
        # pred=3 < label=5 with '<' → clipped to 5; pred=6 stays; pred=5 stays
        np.testing.assert_array_equal(result, [5.0, 6.0, 5.0])

    def test_greater_than_censor_clips_downward(self):
        """If censor is '>', predictions above the label are clipped *down* to the label."""
        labels = np.array([5.0, 5.0, 5.0])
        predictions = np.array([7.0, 4.0, 5.0])
        censor = [">", ">", ">"]
        result = process_censor(labels, predictions, censor)
        # pred=7 > label=5 with '>' → clipped to 5; pred=4 stays; pred=5 stays
        np.testing.assert_array_equal(result, [5.0, 4.0, 5.0])

    def test_equal_censor_no_change(self):
        """If censor is '=', predictions are untouched."""
        labels = np.array([5.0, 5.0])
        predictions = np.array([3.0, 7.0])
        censor = ["=", "="]
        result = process_censor(labels, predictions, censor)
        np.testing.assert_array_equal(result, predictions)

    def test_mixed_censor(self):
        labels = np.array([5.0, 5.0, 5.0])
        predictions = np.array([3.0, 7.0, 5.0])
        censor = ["<", ">", "="]
        result = process_censor(labels, predictions, censor)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_2d_inputs_are_handled(self):
        """process_censor should work when inputs are (N, 1) arrays."""
        labels = np.array([[5.0], [5.0]])
        predictions = np.array([[3.0], [7.0]])
        censor = ["<", ">"]
        result = process_censor(labels, predictions, censor)
        np.testing.assert_array_equal(result, [5.0, 5.0])


# =========================================================================
# process_regression
# =========================================================================


class TestProcessRegression:
    """Tests for the regression metrics aggregation."""

    def test_output_keys(self, regression_labels, regression_predictions):
        output = process_regression(regression_labels, regression_predictions)
        expected_keys = {
            "R2",
            "RMSE",
            "MAE",
            "SPEARMAN_R",
            "Within2Fold",
            "Within3Fold",
        }
        assert set(output.keys()) == expected_keys

    def test_perfect_predictions(self):
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        output = process_regression(labels, labels.copy())
        assert output["R2"] == 1.0
        assert output["RMSE"] == 0.0
        assert output["MAE"] == 0.0
        assert output["SPEARMAN_R"] == 1.0

    def test_log10_mode(self):
        """Smoke test that log10=True path runs without error."""
        labels = np.array([10.0, 100.0, 1000.0, 10000.0])
        preds = np.array([12.0, 90.0, 1100.0, 9500.0])
        output = process_regression(labels, preds, log10=True)
        assert "Within2Fold" in output
        assert "Within3Fold" in output

    def test_nan_labels_are_filtered(self):
        labels = np.array([1.0, np.nan, 3.0])
        preds = np.array([1.0, 2.0, 3.0])
        output = process_regression(labels, preds)
        # After filtering NaN the two remaining points are perfect
        assert output["R2"] == 1.0

    def test_2d_column_inputs(self):
        """Handles (N,1) shaped arrays gracefully."""
        labels = np.array([[1.0], [2.0], [3.0], [4.0]])
        preds = np.array([[1.0], [2.0], [3.0], [4.0]])
        output = process_regression(labels, preds)
        assert output["R2"] == 1.0


# =========================================================================
# process_classification
# =========================================================================


class TestProcessClassification:
    """Tests for classification metrics aggregation."""

    def test_output_keys(
        self,
        classification_labels,
        classification_predictions,
        classification_probabilities,
    ):
        output = process_classification(
            classification_labels,
            classification_predictions,
            classification_probabilities,
        )
        expected_keys = {
            "balanced_accuracy",
            "matthews_corrcoef",
            "f1_score",
            "cohen_kappa",
            "precision",
            "recall",
            "roc_auc",
            "pr_auc",
            "ef_10",
        }
        assert set(output.keys()) == expected_keys

    def test_perfect_classification(self):
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        preds = labels.copy()
        probs = np.array([0.1, 0.05, 0.15, 0.9, 0.95, 0.85])
        output = process_classification(labels, preds, probs)
        assert output["balanced_accuracy"] == 1.0
        assert output["f1_score"] == 1.0

    def test_does_not_mutate_inputs(
        self,
        classification_labels,
        classification_predictions,
        classification_probabilities,
    ):
        labels_copy = classification_labels.copy()
        preds_copy = classification_predictions.copy()
        probs_copy = classification_probabilities.copy()
        process_classification(
            classification_labels,
            classification_predictions,
            classification_probabilities,
        )
        np.testing.assert_array_equal(classification_labels, labels_copy)
        np.testing.assert_array_equal(classification_predictions, preds_copy)
        np.testing.assert_array_equal(classification_probabilities, probs_copy)
