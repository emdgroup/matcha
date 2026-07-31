"""Pydantic schemas for PyTorch-based model architectures in matcha."""

import pydantic as pyd
from typing import Annotated, Literal, Sequence
from matcha.utils.schemas.generic_models import (
    ClassicMatchaModel,
    GraphMixin,
    GINMixin,
    AttentiveFPMixin,
    MPNNMixin,
    GatedGCNMixin,
    E3GNNMixin,
    GPSMixin,
    GPS3DMixin,
    GTMixin,
    GT3DMixin,
    CLMMixin,
    CNNMixin,
    RNNMixin,
    RoFormerMixin,
    TabularMixin,
    MLPMixin,
    SNNMixin,
    FinetunerMixin,
    ChempropFinetunerMixin,
)
from matcha.utils.schemas.base import BaseDataModel


class GINInputModel(ClassicMatchaModel, GraphMixin, GINMixin):
    """Schema for the Graph Isomorphism Network (GIN) model configuration."""

    torch_type: Literal["gin"] = "gin"


class AttentiveFPInputModel(ClassicMatchaModel, GraphMixin, AttentiveFPMixin):
    """Schema for the AttentiveFP graph attention model configuration."""

    torch_type: Literal["attentivefp"] = "attentivefp"
    enc_activation: str | None = pyd.Field(
        default=None,
        exclude=True,
        description="Not used. AttentiveFP uses fixed activations following the original paper.",
    )


class MPNNInputModel(ClassicMatchaModel, GraphMixin, MPNNMixin):
    """Schema for the Message Passing Neural Network (MPNN) model configuration."""

    torch_type: Literal["mpnn"] = "mpnn"


class GatedGCNInputModel(ClassicMatchaModel, GraphMixin, GatedGCNMixin):
    """Schema for the Gated Graph Convolutional Network model configuration."""

    torch_type: Literal["gatedgcn"] = "gatedgcn"


class GPSInputModel(ClassicMatchaModel, GraphMixin, GPSMixin):
    """Schema for the General Powerful Scalable (GPS) graph transformer configuration."""

    torch_type: Literal["gps"] = "gps"


class E3GNNInputModel(ClassicMatchaModel, GraphMixin, E3GNNMixin):
    """Schema for the E(3)-equivariant Graph Neural Network model configuration."""

    torch_type: Literal["e3gnn"] = "e3gnn"


class GPS3DInputModel(ClassicMatchaModel, GraphMixin, GPS3DMixin):
    """Schema for the 3D GPS graph transformer model configuration."""

    torch_type: Literal["gps3d"] = "gps3d"


class GT3DInputModel(ClassicMatchaModel, GraphMixin, GT3DMixin):
    """Schema for the 3D Graph Transformer model configuration."""

    torch_type: Literal["gt3d"] = "gt3d"


class GTInputModel(ClassicMatchaModel, GraphMixin, GTMixin):
    """Schema for the Graph Transformer model configuration."""

    torch_type: Literal["gt"] = "gt"


class CNNInputModel(ClassicMatchaModel, CLMMixin, CNNMixin):
    """Schema for the 1D Convolutional Neural Network model on character sequences."""

    torch_type: Literal["cnn"] = "cnn"


class RoFormerInputModel(ClassicMatchaModel, CLMMixin, RoFormerMixin):
    """Schema for the RoFormer (rotary position embedding transformer) model configuration."""

    torch_type: Literal["roformer"] = "roformer"


class RNNInputModel(ClassicMatchaModel, CLMMixin, RNNMixin):
    """Schema for the Recurrent Neural Network model on character sequences."""

    torch_type: Literal["rnn"] = "rnn"


class MLPInputModel(ClassicMatchaModel, TabularMixin, MLPMixin):
    """Schema for the Multi-Layer Perceptron model on tabular features."""

    torch_type: Literal["mlp"] = "mlp"


class SNNInputModel(ClassicMatchaModel, TabularMixin, SNNMixin):
    """Schema for the Self-Normalizing Neural Network model on tabular features."""

    torch_type: Literal["snn"] = "snn"


class FinetunerInputModel(FinetunerMixin):
    """Schema for the pretrained model finetuning configuration."""

    torch_type: Literal["finetuner"] = "finetuner"


class ChempropFinetunerInputModel(ChempropFinetunerMixin):
    """Schema for the Chemprop pretrained model finetuning configuration."""

    torch_type: Literal["chemprop_finetuner"] = "chemprop_finetuner"


class ChempropInputModel(BaseDataModel):
    """Schema for the Chemprop directed message passing neural network configuration."""

    torch_type: Literal["chemprop"] = "chemprop"
    enc_atom_hidden_dim: int
    enc_num_layers: int
    enc_dropout: float
    enc_activation: str
    enc_readout: str
    additional_mol_features_dim: int
    pred_hidden_dim: int
    pred_num_layers: int
    pred_dropout: float
    pred_activation: str
    num_endpoints: int
    loss_fn: str
    optimizer: str
    optimizer_args: dict
    scheduler: str
    scheduler_args: dict


TorchModel = (
    GINInputModel
    | AttentiveFPInputModel
    | MPNNInputModel
    | GatedGCNInputModel
    | E3GNNInputModel
    | GPSInputModel
    | GPS3DInputModel
    | GTInputModel
    | GT3DInputModel
    | ChempropInputModel
    | CNNInputModel
    | RNNInputModel
    | RoFormerInputModel
    | MLPInputModel
    | SNNInputModel
    | FinetunerInputModel
    | ChempropFinetunerInputModel
)


class TorchModels(BaseDataModel):
    """Container schema for a sequence of discriminated PyTorch model configurations."""

    inputmodels: Sequence[Annotated[TorchModel, pyd.Field(discriminator="torch_type")]]
