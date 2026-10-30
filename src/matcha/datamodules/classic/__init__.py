"""Classic datamodules for molecular featurization (graph, tabular, CLM, combined)."""

from typing import Any

# NOTE: DataModule classes are imported lazily via ``__getattr__`` to avoid a
# circular import.  The cycle is:
#
#   base_datamodule  ->  classic.label_encoder
#                        (triggers classic/__init__)
#   classic/__init__  ->  tabular_datamodule  ->  base_datamodule  (CIRCULAR)
#
# Only ``label_encoder``, ``label_transform``, and ``rdkit_engine`` are
# imported eagerly because they do NOT depend on ``base_datamodule``.

from matcha.datamodules.classic.label_encoder import (
    BaseLabelEncoder,
    BinaryClassificationLabelEncoder,
    RegressionLabelEncoder,
    LabelEncoderRegistry,
)
from matcha.datamodules.classic.label_transform import LabelTransform
from matcha.datamodules.classic.rdkit_engine import Engine

# Lazy-import mapping: attribute name -> (module path, object name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TabularDataModule": (
        "matcha.datamodules.classic.tabular_datamodule",
        "TabularDataModule",
    ),
    "RDKIT_FEATURES": (
        "matcha.datamodules.classic.tabular_datamodule",
        "RDKIT_FEATURES",
    ),
    "GraphDataModule": (
        "matcha.datamodules.classic.graph_datamodule",
        "GraphDataModule",
    ),
    "Graph3DDataModule": (
        "matcha.datamodules.classic.graph_datamodule",
        "Graph3DDataModule",
    ),
    "ATOM_FEAT_DIM": ("matcha.datamodules.classic.graph_datamodule", "ATOM_FEAT_DIM"),
    "BOND_FEAT_DIM": ("matcha.datamodules.classic.graph_datamodule", "BOND_FEAT_DIM"),
    "CLMDataModule": ("matcha.datamodules.classic.clm_datamodule", "CLMDataModule"),
    "CombinedDataModule": (
        "matcha.datamodules.classic.combined_datamodule",
        "CombinedDataModule",
    ),
    "default_merge": (
        "matcha.datamodules.classic.combined_datamodule",
        "default_merge",
    ),
    "ChempropDataModule": (
        "matcha.datamodules.classic.chemprop_datamodule",
        "ChempropDataModule",
    ),
    "GraphPE": ("matcha.datamodules.classic.graph_positional_encoder", "GraphPE"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, obj_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, obj_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Label encoding
    "BaseLabelEncoder",
    "BinaryClassificationLabelEncoder",
    "RegressionLabelEncoder",
    "LabelEncoderRegistry",
    "LabelTransform",
    # DataModules
    "TabularDataModule",
    "GraphDataModule",
    "Graph3DDataModule",
    "CLMDataModule",
    "CombinedDataModule",
    "ChempropDataModule",
    # Utilities
    "GraphPE",
    "Engine",
    "default_merge",
    # Constants
    "RDKIT_FEATURES",
    "ATOM_FEAT_DIM",
    "BOND_FEAT_DIM",
]
