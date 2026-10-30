"""Manager classes that encapsulate cross-cutting concerns for sklearn-compatible models."""

from matcha.sklearn.managers.datamodule_manager import DataModuleManager
from matcha.sklearn.managers.mlflow_manager import MLFlowManager
from matcha.sklearn.managers.serialization_manager import SerializationManager
from matcha.sklearn.managers.training_manager import (
    TrainingManager,
    CLMTrainingManager,
    FinetunerTrainingManager,
)
from matcha.sklearn.managers.uncertainty_manager import UncertaintyManager
from matcha.sklearn.managers.explainability_manager import ExplainabilityManager
from matcha.sklearn.managers.hpo_manager import HPOManager
from matcha.sklearn.managers.ensemble_mlflow_manager import EnsembleMLFlowManager
from matcha.sklearn.managers.ensemble_serialization_manager import (
    EnsembleSerializationManager,
)
from matcha.sklearn.managers.ensemble_calibration_manager import (
    EnsembleCalibrationManager,
)

__all__ = [
    "DataModuleManager",
    "MLFlowManager",
    "SerializationManager",
    "TrainingManager",
    "CLMTrainingManager",
    "FinetunerTrainingManager",
    "UncertaintyManager",
    "ExplainabilityManager",
    "HPOManager",
    "EnsembleMLFlowManager",
    "EnsembleSerializationManager",
    "EnsembleCalibrationManager",
]
