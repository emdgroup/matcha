"""Pydantic validation schemas for matcha model configurations, datamodules, and data.

This package provides input validation models for all matcha components including
PyTorch model architectures, scikit-learn wrappers, datamodule configurations,
molecular datasets, explainability, and calibration settings.
"""

# Torch API schemas
from matcha.utils.schemas.torch_api import (
    GINInputModel,
    AttentiveFPInputModel,
    MPNNInputModel,
    GatedGCNInputModel,
    E3GNNInputModel,
    E3GNNPretrainingInputModel,
    GPSInputModel,
    GPS3DInputModel,
    GTInputModel,
    GT3DInputModel,
    CNNInputModel,
    RNNInputModel,
    RoFormerInputModel,
    MLPInputModel,
    SNNInputModel,
    FinetunerInputModel,
    ChempropFinetunerInputModel,
    ChempropInputModel,
)

# Datamodule schemas
from matcha.utils.schemas.datamodules import (
    ChempropDataModuleInputModel,
    TabularDataModuleInputModel,
    GraphDataModuleInputModel,
    GraphPretrainingDataModuleInputModel,
    Graph3DDataModuleInputModel,
    Graph3DPretrainingDataModuleInputModel,
    CLMDataModuleInputModel,
    CombinedDataModuleInputModel,
)

# Sklearn API schemas
from matcha.utils.schemas.sklearn_api import (
    ScikitLearnInputModel,
    ScikitLearnEnsembleInputModel,
)

# Data schemas
from matcha.utils.schemas.data import MolDataset, MolReadout

# Explainability schemas
from matcha.utils.schemas.explainability import ExplainerInputModel

# Calibration schemas
from matcha.utils.schemas.calibration import (
    ICPBaseInputModel,
    ICPRegressionInputModel,
    ICPClassificationInputModel,
    EMBaseInputModel,
    EMRegressionInputModel,
    EMClassificationInputModel,
)

__all__ = [
    # Torch API
    "GINInputModel",
    "AttentiveFPInputModel",
    "MPNNInputModel",
    "GatedGCNInputModel",
    "E3GNNInputModel",
    "E3GNNPretrainingInputModel",
    "GPSInputModel",
    "GPS3DInputModel",
    "GTInputModel",
    "GT3DInputModel",
    "CNNInputModel",
    "RNNInputModel",
    "RoFormerInputModel",
    "MLPInputModel",
    "SNNInputModel",
    "FinetunerInputModel",
    "ChempropFinetunerInputModel",
    "ChempropInputModel",
    # Datamodules
    "ChempropDataModuleInputModel",
    "TabularDataModuleInputModel",
    "GraphDataModuleInputModel",
    "GraphPretrainingDataModuleInputModel",
    "Graph3DDataModuleInputModel",
    "Graph3DPretrainingDataModuleInputModel",
    "CLMDataModuleInputModel",
    "CombinedDataModuleInputModel",
    # Sklearn API
    "ScikitLearnInputModel",
    "ScikitLearnEnsembleInputModel",
    # Data
    "MolDataset",
    "MolReadout",
    # Explainability
    "ExplainerInputModel",
    # Calibration
    "ICPBaseInputModel",
    "ICPRegressionInputModel",
    "ICPClassificationInputModel",
    "EMBaseInputModel",
    "EMRegressionInputModel",
    "EMClassificationInputModel",
]
