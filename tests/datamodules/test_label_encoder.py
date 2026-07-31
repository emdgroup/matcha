"""Tests for LabelEncoder classes."""

import numpy as np
import pandas as pd

from matcha.datamodules.classic.label_encoder import (
    RegressionLabelEncoder,
    BinaryClassificationLabelEncoder,
    LabelEncoderRegistry,
)


# ===================================================================
# Registry
# ===================================================================


class TestLabelEncoderRegistry:
    def test_regression_registered(self):
        encoder = LabelEncoderRegistry["regression"]()
        assert isinstance(encoder, RegressionLabelEncoder)

    def test_binary_classification_registered(self):
        encoder = LabelEncoderRegistry["binary_classification"]()
        assert isinstance(encoder, BinaryClassificationLabelEncoder)


# ===================================================================
# RegressionLabelEncoder
# ===================================================================


class TestRegressionLabelEncoderInit:
    def test_empty_init(self):
        enc = RegressionLabelEncoder()
        assert enc.params.task_labels == {}

    def test_init_with_params(self):
        enc = RegressionLabelEncoder({0: {"task_label": "logS"}})
        assert enc.params.task_labels[0] == "logS"

    def test_init_multiple_tasks(self):
        enc = RegressionLabelEncoder(
            {0: {"task_label": "logS"}, 1: {"task_label": "logP"}}
        )
        assert len(enc.params.task_labels) == 2
        assert enc.params.task_labels[1] == "logP"


class TestRegressionLabelEncoderProcess:
    def test_process_returns_dataframe(self):
        enc = RegressionLabelEncoder({0: {"task_label": "logS"}})
        preds = np.random.rand(10, 1)
        result = enc.process(preds, tag="preds")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 1)

    def test_process_column_naming(self):
        enc = RegressionLabelEncoder({0: {"task_label": "logS"}})
        preds = np.random.rand(5, 1)
        result = enc.process(preds, tag="test")
        assert "logS_test" in result.columns

    def test_process_multitask(self):
        enc = RegressionLabelEncoder(
            {0: {"task_label": "logS"}, 1: {"task_label": "logP"}}
        )
        preds = np.random.rand(5, 2)
        result = enc.process(preds, tag="preds")
        assert result.shape == (5, 2)
        assert "logS_preds" in result.columns
        assert "logP_preds" in result.columns


class TestRegressionLabelEncoderIsSet:
    def test_is_set_returns_false(self):
        enc = RegressionLabelEncoder()
        assert enc.is_set() is False


class TestRegressionLabelEncoderStateDict:
    def test_state_dict_roundtrip(self):
        enc = RegressionLabelEncoder(
            {0: {"task_label": "logS"}, 1: {"task_label": "logP"}}
        )
        state = enc.state_dict()
        enc2 = RegressionLabelEncoder()
        enc2.load_state_dict(state)
        assert enc2.params.task_labels == enc.params.task_labels


class TestRegressionLabelEncoderSetTaskParams:
    def test_set_task_params(self):
        enc = RegressionLabelEncoder()
        enc._set_task_params(0, "logS")
        assert enc.params.task_labels[0] == "logS"


# ===================================================================
# BinaryClassificationLabelEncoder
# ===================================================================


class TestBinaryClassificationEncoderInit:
    def test_empty_init(self):
        enc = BinaryClassificationLabelEncoder()
        assert enc.params.task_labels == {}
        assert enc.params.class_thresholds == {}

    def test_init_with_params(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        assert enc.params.task_labels[0] == "activity"
        assert enc.params.class_thresholds[0] == [0.5]
        assert enc.params.class_labels[0] == ["inactive", "active"]


class TestBinaryClassificationEncoderIsSet:
    def test_is_set_with_thresholds(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        assert enc.is_set() is True

    def test_is_set_without_thresholds(self):
        enc = BinaryClassificationLabelEncoder()
        assert enc.is_set() is False


class TestBinaryClassificationEncoderContinuousToCategorical:
    def test_basic_thresholding(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        values = np.array([0.1, 0.6, 0.3, 0.9, 0.5])
        result = enc._continuous_to_categorical(values, 0)
        expected = np.array([[0], [1], [0], [1], [0]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_nan_handling(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        values = np.array([0.1, np.nan, 0.9])
        result = enc._continuous_to_categorical(values, 0)
        assert np.isnan(result[1, 0])


class TestBinaryClassificationEncoderIntegerToLabel:
    def test_basic_conversion(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        matrix = np.array([[0], [1], [0]])
        result = enc._integer_to_label(matrix)
        assert result == [["inactive", "active", "inactive"]]


class TestBinaryClassificationEncoderProcess:
    def test_process_returns_dataframe(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        preds = np.array([[0], [1], [0]])
        result = enc.process(preds, tag="clf")
        assert isinstance(result, pd.DataFrame)
        assert "activity_clf" in result.columns

    def test_process_with_continuous_probabilities(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        preds = np.array([[0.2], [0.51423], [0.8], [0.49]])
        result = enc.process(preds, tag="label")
        assert result["activity_label"].tolist() == [
            "inactive",
            "active",
            "active",
            "inactive",
        ]

    def test_process_without_label_conversion(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        preds = np.array([[0], [1], [0]])
        result = enc.process(preds, tag="clf", convert_to_labels=False)
        assert isinstance(result, pd.DataFrame)


class TestBinaryClassificationEncoderAllToCategorical:
    def test_single_task(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        values = np.array([[0.1], [0.9]])
        result = enc._all_to_categorical(values)
        assert result.shape == (2, 1)

    def test_multitask(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "task0",
                    "class_thresholds": [0.5],
                    "class_labels": ["low", "high"],
                },
                1: {
                    "task_label": "task1",
                    "class_thresholds": [0.3],
                    "class_labels": ["no", "yes"],
                },
            }
        )
        values = np.array([[0.1, 0.4], [0.9, 0.2]])
        result = enc._all_to_categorical(values)
        assert result.shape == (2, 2)


class TestBinaryClassificationEncoderStateDict:
    def test_state_dict_roundtrip(self):
        enc = BinaryClassificationLabelEncoder(
            {
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                }
            }
        )
        state = enc.state_dict()
        enc2 = BinaryClassificationLabelEncoder()
        enc2.load_state_dict(state)
        assert enc2.params.task_labels == enc.params.task_labels
        assert enc2.params.class_thresholds == enc.params.class_thresholds
