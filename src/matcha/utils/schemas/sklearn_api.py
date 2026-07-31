"""Pydantic schemas for scikit-learn-compatible matcha model configurations."""

from matcha.utils.schemas.base import BaseDataModel
from typing import Dict, Literal, Optional, Any
from matcha.utils.schemas.datamodules import DataModuleModel
from matcha.utils.schemas.torch_api import TorchModel
from matcha.utils.schemas.calibration import CalibratorModel


class TrainingInputModel(BaseDataModel):
    """Schema for training loop hyperparameters such as epochs, batch size, and early stopping."""

    num_epochs: int = 100
    batch_size: int = 32
    stochastic_weight_averaging: bool = False
    accelerator: str = "auto"
    devices: int = 1
    early_stopping: bool = True
    patience: int = 10
    seed: int = 0


class MLFlowInputModel(BaseDataModel):
    """Schema for MLflow tracking configuration including experiment name and server URI."""

    experiment: str
    run: Optional[str] = None
    tag: Optional[Dict[str, Any]] = None
    log_dir: str = "./matcha_log"
    server_uri: Optional[str] = None


class RegressionInputModel(BaseDataModel):
    """Schema that specifies a regression task type."""

    task_type: Literal["regression"] = "regression"


class MetadataInputModel(BaseDataModel):
    """Schema for model metadata including ownership, versioning, and description."""

    model_type: str
    model_version: int
    model_name: str
    model_owner: str
    model_scope: str
    matcha_version: str
    date: str
    description: str
    extra: dict[str, Any] = {}


class TuningInputModel(BaseDataModel):
    """Schema for hyperparameter tuning configuration including search budgets and grids."""

    architecture_search_budget: int
    architecture_grid: dict | None
    optimizer_search_budget: int
    optimizer_grid: dict | None
    scheduler_grid: dict | None


class ScikitLearnInputModel(BaseDataModel):
    """Top-level schema for a single scikit-learn-compatible matcha model configuration.

    Combines training, datamodule, model architecture, metadata, and optional
    calibration and tuning settings.
    """

    training: TrainingInputModel
    datamodule: DataModuleModel
    model: TorchModel
    metadata: MetadataInputModel
    task_type: Literal["regression", "binary_classification"]
    calibration: CalibratorModel | None
    mlflow: MLFlowInputModel | None
    tuning: TuningInputModel | None


class ScikitLearnEnsembleInputModel(BaseDataModel):
    """Schema for an ensemble of scikit-learn-compatible matcha models.

    Defines the ensemble architecture, base learner configuration, and number of models.
    """

    architecture: str
    learner: ScikitLearnInputModel | dict
    n_models: int
    calibration: CalibratorModel | None
    mlflow: MLFlowInputModel | None
    metadata: MetadataInputModel
    seed: int = 0
