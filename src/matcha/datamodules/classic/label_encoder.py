"""Label encoders for converting model outputs to human-readable formats."""

import numpy as np
import pandas as pd
from matcha.utils.registry import ClassRegistry
from matcha.utils.schemas.label import (
    RegressionEncoderInputModel,
    BinaryClassificationEncoderInputModel,
)

LabelEncoderRegistry = ClassRegistry()


class BaseLabelEncoder:
    """Converts predictions into human-readable outputs.

    This base class wraps output arrays from models into dataframes
    with meaningful, user-defined column names.

    Example usage:

    .. code-block:: python
        encoder = RegressionLabelEncoder({
            0: {"task_label": "LOGS_THERMO"},
            1: {"task_label": "LOGS"}
                })
        preds = np.random.rand(1000,2)
        output = encoder.process(preds, tag="predictions")

    :param dict params: ruleset to use when processing arrays, see
        example above
    """

    def __init__(self, params: dict = {}):
        # Configure from params if provided
        if params:
            self._configure_from_params(params)

    def is_set(self) -> bool:
        """Check whether the encoder has classification thresholds configured.

        :returns: False for the base class; overridden by subclasses
        :rtype: bool
        """
        return False

    @property
    def label_names(self) -> str | list[str]:
        return list(self.params.task_labels.values())

    def _configure_from_params(self, params: dict):
        """Configure the label encoder from initialization parameters"""
        params = params.copy()

        # Remove task_type as it's handled separately
        params.pop("encoder_type", None)

        # Check if this is a state dict format (direct assignment)
        if self._is_state_dict_format(params):
            # Use the same logic as load_state_dict
            self.load_state_dict(params)
        else:
            # This is initialization format (numbered tasks)
            self._load_from_init_format(params)

    def _is_state_dict_format(self, params: dict) -> bool:
        """Check if params is in state dict format vs initialization format"""
        state_dict_keys = {
            "task_labels",
            "class_thresholds",
            "class_labels",
            "num_classes",
        }
        return any(key in params for key in state_dict_keys)

    def _load_from_init_format(self, params: dict):
        """Load from initialization format (numbered tasks)"""
        allowed_keys = {"task_label", "class_thresholds", "class_labels"}

        for task_id_str, task_params in params.items():
            try:
                task_id = int(task_id_str)
            except ValueError:
                raise ValueError(f"Task ID must be an integer, got: {task_id_str}")

            unexpected_keys = set(task_params.keys()) - allowed_keys
            if unexpected_keys:
                raise ValueError(
                    f"Unexpected keys found in params[{task_id}]: {unexpected_keys}"
                )

            self._set_task_params(task_id, **task_params)

    def _set_task_params(
        self,
        task_idx: int = 0,
        task_label: str = "output",
        class_thresholds: list[float] | None = None,
        class_labels: list[str] | None = None,
    ):
        """Sets ruleset for conversion for a specific task"""
        self.params.task_labels[task_idx] = task_label

    def _array_to_dataframe(
        self, input: np.ndarray | list, tag: list[str]
    ) -> pd.DataFrame:
        """Wraps predictions into dataframe, excluding dummy columns"""

        labels = list(self.params.task_labels.values())
        labels = [f"{x}_{tag}" for x in labels]

        # Exclude columns where label contains 'dummy'
        filtered = [
            (label, idx) for idx, label in enumerate(labels) if "dummy" not in label
        ]

        if isinstance(input, np.ndarray):
            # Filter columns for ndarray
            filtered_indices = [idx for _, idx in filtered]
            filtered_labels = [label for label, _ in filtered]
            filtered_input = input[:, filtered_indices] if input.ndim > 1 else input
            return pd.DataFrame(filtered_input, columns=filtered_labels)
        elif isinstance(input, list):
            # Filter columns for list
            filtered_labels = [label for label, _ in filtered]
            filtered_input = [input[idx] for _, idx in filtered]
            return pd.DataFrame(
                {
                    label: sublist
                    for label, sublist in zip(filtered_labels, filtered_input)
                }
            )

    def state_dict(self) -> dict:
        """Return the complete internal state for serialization"""
        return self.params.model_dump()

    def load_state_dict(self, state_dict: dict):
        """Load the complete internal state from serialization"""
        self.params = self._input_model.model_validate(state_dict)

    @classmethod
    def create_empty(cls):
        """Create an empty label encoder that can be populated via load_state_dict"""
        return cls({})


@LabelEncoderRegistry.register(alias="regression")
class RegressionLabelEncoder(BaseLabelEncoder):
    """Label encoder for regression tasks.

    Wraps raw prediction arrays into labelled DataFrames with user-defined
    column names.
    """

    def __init__(self, params: dict = {}):
        self.params = RegressionEncoderInputModel(
            task_labels={},
        )
        self._input_model = RegressionEncoderInputModel
        super().__init__(params)

    def process(self, input: np.ndarray, tag: str, convert_to_labels: bool = True):
        """Converts the prediction array into a labelled dataframe,
        dropping any dummy columns."""
        output = [input[:, i] for i in range(input.shape[1])]
        return self._array_to_dataframe(output, tag)


@LabelEncoderRegistry.register(alias="binary_classification")
class BinaryClassificationLabelEncoder(BaseLabelEncoder):
    """Label encoder for binary classification tasks.

    Converts continuous predictions into categorical labels using
    user-defined thresholds.
    """

    def __init__(self, params: dict = {}):
        self.params = BinaryClassificationEncoderInputModel(
            task_labels={},
            class_thresholds={},
            class_labels={},
            num_classes={},
        )
        self._input_model = BinaryClassificationEncoderInputModel
        super().__init__(params)

    def is_set(self) -> bool:
        return bool(self.params.class_thresholds)

    def _set_task_params(
        self,
        task_idx: int = 0,
        task_label: str = "output",
        class_thresholds: list[float] | None = None,
        class_labels: list[str] | None = None,
    ):
        """Sets ruleset for conversion for a specific task"""
        self.params.task_labels[task_idx] = task_label
        self.params.class_thresholds[task_idx] = class_thresholds
        self.params.class_labels[task_idx] = class_labels
        if class_labels is not None:
            self.params.num_classes[task_idx] = len(class_labels)
        else:
            self.params.num_classes[task_idx] = None

    def _continuous_to_categorical(self, input: np.ndarray, task_idx: int):
        """Chops continuous predictions into bins, turning them into a
        one-hot-encoded matrix"""

        output = np.zeros((input.shape[0], 1))
        idx_1 = np.where(input > self.params.class_thresholds[task_idx])[0]
        idx_nan = np.isnan(input)
        output[idx_1] = 1
        if any(np.isnan(input)):
            idx_nan = np.isnan(input)
            output[idx_nan] = np.nan

        return output

    def _integer_to_label(self, encoded_matrix: np.ndarray):
        """Converts a binary matrix into a list of labels, depending
        on where the column with 1 is in the matrix"""

        output = []

        for col_idx in range(encoded_matrix.shape[1]):  # For each column (K columns)
            column_labels = []

            for row_idx in range(encoded_matrix.shape[0]):  # For each row (N rows)
                binary_value = int(encoded_matrix[row_idx, col_idx])
                label = self.params.class_labels[col_idx][binary_value]
                column_labels.append(label)

            output.append(column_labels)

        return output

    def _all_to_categorical(self, input: np.ndarray):
        out = []
        if input.ndim > 1:
            for i in range(input.shape[1]):
                out.append(self._continuous_to_categorical(input[:, i], i))
        else:
            out.append(self._continuous_to_categorical(input, 0))
        return np.concatenate(out, axis=1)

    def process(self, input: np.ndarray, tag: str, convert_to_labels: bool = True):
        """Executes the pipeline end-to-end"""

        if convert_to_labels:
            categorical = self._all_to_categorical(input)
            output = self._integer_to_label(categorical)
        else:
            output = input.copy()
        return self._array_to_dataframe(output, tag)
