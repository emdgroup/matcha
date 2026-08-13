"""Pydantic schemas for matcha datamodule configurations."""

from enum import Enum
from typing import Annotated, Callable, Dict, Literal, Optional, Sequence

from pydantic import Field, field_validator
from matcha.utils.schemas.base import BaseDataModel
from matcha.utils.schemas.label import (
    RegressionEncoderInputModel,
    BinaryClassificationEncoderInputModel,
    LabelTransformInputModel,
)


class HandleMissing(Enum):
    """Enum class to handle missing values in the data."""

    RAISE = "raise"
    FILL = "fill"


class BaseDataModuleFields(BaseDataModel):
    """Base fields common to all datamodule types."""

    is_classification: bool = False
    scaler_type: Literal["standard", "quantile"] = "standard"
    clip: bool = True
    label_encoder_params: (
        dict | RegressionEncoderInputModel | BinaryClassificationEncoderInputModel
    ) = {}
    label_transform_params: dict | LabelTransformInputModel = {}
    batch_size: int = 256
    num_workers: int = 0
    augment_resonance: bool = False


class TabularDataModuleInputModel(BaseDataModuleFields):
    """Schema for tabular feature-based datamodule configuration."""

    datamodule_type: Literal["tabular"] = "tabular"

    # Tabular-specific fields
    input_dim: int
    feature_list: list[str]
    engine_params: Optional[Dict] = None


class CLMDataModuleInputModel(BaseDataModuleFields):
    """Schema for character-level model datamodule with SMILES augmentation settings."""

    datamodule_type: Literal["clm"] = "clm"

    # CLM-specific fields
    max_length: Annotated[int, Field(gt=0)] = 200
    num_augmentations: Annotated[int, Field(ge=0)] = 3
    num_test_augmentations: Annotated[int, Field(ge=0)] = 0
    include_canonical: bool = True
    dictionary: dict = {"pad": 0, "unk": 1, "cls": 2, "mask": 3}
    num_tokens: int = 4


class CLMMLMDataModuleInputModel(CLMDataModuleInputModel):
    """Schema for masked language model datamodule extending CLM with a mask rate."""

    datamodule_type: Literal["clm_mlm"] = "clm_mlm"

    # MLM-specific field
    mask_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15


class GraphDataModuleInputModel(BaseDataModuleFields):
    """Schema for molecular graph datamodule with positional encoding settings."""

    datamodule_type: Literal["graph"] = "graph"

    # Graph-specific fields
    laplacian_k: Annotated[int, Field(ge=0)] = 10
    rwse_k: Annotated[int, Field(ge=0)] = 20
    rrwp_k: Annotated[int, Field(ge=0)] = 20
    elstatic_k: Annotated[int, Field(ge=0)] = 0
    distmat_k: Annotated[int, Field(ge=0)] = 0
    compute_distances: bool = True
    num_virtual_nodes: Annotated[int, Field(ge=0)] = 0
    init_virtual_nodes: bool = False


class GraphPretrainingDataModuleInputModel(GraphDataModuleInputModel):
    """Schema for graph pretraining datamodule with optional graph/node-level scaling."""

    datamodule_type: Literal["graph_pretraining"] = "graph_pretraining"

    # Pretraining-specific fields
    scale_y_graph: bool = False
    scale_y_node: bool = False


class Graph3DPretrainingDataModuleInputModel(GraphPretrainingDataModuleInputModel):
    """Schema for 3D graph pretraining datamodule with user-supplied coordinates.

    Inherits every field from :class:`GraphPretrainingDataModuleInputModel`. The
    3D variant intentionally omits ``embed_timeout`` — coordinates are supplied
    by the user, never generated on the fly with ETKDG.
    """

    datamodule_type: Literal["graph3d_pretraining"] = "graph3d_pretraining"


class Graph3DDataModuleInputModel(BaseDataModuleFields):
    """Schema for 3D molecular graph datamodule with conformer generation settings."""

    datamodule_type: Literal["graph3d"] = "graph3d"

    # Graph3D-specific fields
    laplacian_k: Annotated[int, Field(ge=0)] = 10
    rwse_k: Annotated[int, Field(ge=0)] = 20
    rrwp_k: Annotated[int, Field(ge=0)] = 20
    elstatic_k: Annotated[int, Field(ge=0)] = 0
    distmat_k: Annotated[int, Field(ge=0)] = 0
    compute_distances: bool = True
    num_virtual_nodes: Annotated[int, Field(ge=0)] = 0
    init_virtual_nodes: bool = False
    embed_timeout: Annotated[float, Field(gt=0.0)] = 30.0


class CombinedDataModuleInputModel(BaseDataModuleFields):
    """Schema for combining multiple datamodules with optional merge functions."""

    model_config = {"arbitrary_types_allowed": True}

    datamodule_type: Literal["combined"] = "combined"

    # Combined-specific fields
    datamodules: list
    merge_fn: Optional[Dict[str, Callable]] = None

    @field_validator("merge_fn")
    @classmethod
    def validate_merge_fn(cls, v):
        if v is not None:
            if not isinstance(v, dict):
                raise TypeError(
                    "merge_fn must be a dictionary mapping strings to callables."
                )
            for key, func in v.items():
                if not isinstance(key, str):
                    raise TypeError("Keys in merge_fn must be strings.")
                if not callable(func):
                    raise TypeError("Values in merge_fn must be callable functions.")
        return v


class ChempropDataModuleInputModel(BaseDataModuleFields):
    """Schema for Chemprop datamodule configuration with optional tabular features."""

    datamodule_type: Literal["chemprop"] = "chemprop"

    # Chemprop-specific fields
    input_dim: int = 0  # Will be calculated based on features
    feature_list: Optional[list[str]] = None
    engine_params: Optional[Dict] = None


DataModuleModel = (
    TabularDataModuleInputModel
    | CLMDataModuleInputModel
    | CLMMLMDataModuleInputModel
    | GraphDataModuleInputModel
    | GraphPretrainingDataModuleInputModel
    | Graph3DDataModuleInputModel
    | Graph3DPretrainingDataModuleInputModel
    | CombinedDataModuleInputModel
    | ChempropDataModuleInputModel
)


class DataModuleModels(BaseDataModel):
    """Container schema for a sequence of discriminated datamodule configurations."""

    inputmodels: Sequence[
        Annotated[DataModuleModel, Field(discriminator="datamodule_type")]
    ]
