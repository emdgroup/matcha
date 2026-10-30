from typing import Annotated, Literal, Sequence

from pydantic import Field
from matcha.utils.schemas.base import BaseDataModel


class LabelTransformInputModel(BaseDataModel):
    """Schema for label transformation settings including mapping and clipping."""

    transform_map: str | list[str] | dict | None
    y_clip: dict | None


class BaseEncoderInputModel(BaseDataModel):
    """Base schema for label encoder configuration with task and class metadata."""

    task_labels: dict = {}
    class_thresholds: dict = {}
    class_labels: dict = {}
    num_classes: dict = {}


class RegressionEncoderInputModel(BaseDataModel):
    """Schema for regression label encoder configuration."""

    task_labels: dict = {}
    encoder_type: Literal["regression"] = "regression"


class BinaryClassificationEncoderInputModel(BaseEncoderInputModel):
    """Schema for binary classification label encoder with thresholds and class labels."""

    encoder_type: Literal["binary_classification"] = "binary_classification"


LabelEncoderModel = RegressionEncoderInputModel | BinaryClassificationEncoderInputModel


class DataModuleModels(BaseDataModel):
    """Container holding a sequence of label encoder models discriminated by encoder type."""

    inputmodels: Sequence[
        Annotated[LabelEncoderModel, Field(discriminator="encoder_type")]
    ]
