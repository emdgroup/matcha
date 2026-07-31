"""Data modules for molecular featurization and dataset preparation."""

from typing import Any

# Base classes and registries
from matcha.datamodules.base_datamodule import BaseDataModule, DataModuleRegistry

# Label encoding – safe to import eagerly (no circular dependency)
from matcha.datamodules.classic.label_encoder import (
    BaseLabelEncoder,
    BinaryClassificationLabelEncoder,
    RegressionLabelEncoder,
    LabelEncoderRegistry,
)
from matcha.datamodules.classic.label_transform import LabelTransform
from matcha.datamodules.classic.rdkit_engine import Engine

# Utilities – no circular dependency
from matcha.datamodules.utils import CombinedStackDataset

# -------------------------------------------------------------------
# Everything below depends (directly or transitively) on base_datamodule
# and is therefore imported lazily to break the circular import chain:
#
#   base_datamodule -> classic.label_encoder -> (triggers classic/__init__)
#   classic/__init__ -> tabular_datamodule -> base_datamodule  (CIRCULAR)
# -------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Classic DataModules
    "TabularDataModule": (
        "matcha.datamodules.classic.tabular_datamodule",
        "TabularDataModule",
    ),
    "GraphDataModule": (
        "matcha.datamodules.classic.graph_datamodule",
        "GraphDataModule",
    ),
    "Graph3DDataModule": (
        "matcha.datamodules.classic.graph_datamodule",
        "Graph3DDataModule",
    ),
    "CLMDataModule": ("matcha.datamodules.classic.clm_datamodule", "CLMDataModule"),
    "CombinedDataModule": (
        "matcha.datamodules.classic.combined_datamodule",
        "CombinedDataModule",
    ),
    "ChempropDataModule": (
        "matcha.datamodules.classic.chemprop_datamodule",
        "ChempropDataModule",
    ),
    # Pretraining DataModules
    "CLMMLMDataModule": (
        "matcha.datamodules.pretraining.clm_mlm_datamodule",
        "CLMMLMDataModule",
    ),
    "OnTheFlyDataModule": (
        "matcha.datamodules.pretraining.on_the_fly_datamodule",
        "OnTheFlyDataModule",
    ),
    "OnTheFlyMLMDataModule": (
        "matcha.datamodules.pretraining.on_the_fly_mlm_datamodule",
        "OnTheFlyMLMDataModule",
    ),
    "GraphPretrainingDataModule": (
        "matcha.datamodules.pretraining.graph_pretraining_datamodule",
        "GraphPretrainingDataModule",
    ),
    "OnTheFlyGraphPretrainingDataModule": (
        "matcha.datamodules.pretraining.on_the_fly_graph_pretraining_datamodule",
        "OnTheFlyGraphPretrainingDataModule",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, obj_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, obj_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "BaseDataModule",
    "DataModuleRegistry",
    "BaseLabelEncoder",
    "LabelEncoderRegistry",
    "LabelTransform",
    # Label encoders
    "BinaryClassificationLabelEncoder",
    "RegressionLabelEncoder",
    # Classic DataModules
    "TabularDataModule",
    "GraphDataModule",
    "Graph3DDataModule",
    "CLMDataModule",
    "CombinedDataModule",
    "ChempropDataModule",
    # Pretraining DataModules
    "CLMMLMDataModule",
    "OnTheFlyDataModule",
    "OnTheFlyMLMDataModule",
    "GraphPretrainingDataModule",
    "OnTheFlyGraphPretrainingDataModule",
    # Utilities
    "CombinedStackDataset",
    "Engine",
]
