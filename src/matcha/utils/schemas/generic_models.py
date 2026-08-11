from typing import Any
from matcha.utils.schemas.base import BaseDataModel


class ClassicMatchaModel(BaseDataModel):
    """Base schema for all standard matcha models defining shared training parameters."""

    additional_mol_features_dim: int
    num_endpoints: int
    loss_fn: str
    loss_args: dict
    optimizer: str
    optimizer_args: dict[str, Any]
    scheduler: str
    scheduler_args: dict[str, Any]


# ------------------------------------------------------------------------------
#                               GNNs
# ------------------------------------------------------------------------------


class GraphMixin(BaseDataModel):
    """Mixin schema for shared graph neural network encoder parameters."""

    enc_num_layers: int
    enc_readout: str
    pred_hidden_dims: list[int] | None
    enc_atom_input_dim: int
    enc_atom_hidden_dim: int
    enc_activation: str
    enc_dropout: float
    enc_laplacian_k: int
    enc_rwse_k: int
    enc_distmat_k: int
    enc_elstatic_k: int
    pred_activation: str
    pred_dropout: float
    pred_task_head_dims: list[int] | None
    enc_rrwp_k: int


class GINMixin(BaseDataModel):
    """Mixin schema for GIN-specific parameters such as aggregation and jumping knowledge."""

    enc_aggregation: str
    enc_norm: str | None
    enc_jk: str
    enc_eps: float
    enc_train_eps: bool


class AttentiveFPMixin(BaseDataModel):
    """Mixin schema for AttentiveFP-specific parameters."""

    enc_bond_input_dim: int
    enc_jk: str


class MPNNMixin(BaseDataModel):
    """Mixin schema for MPNN-specific parameters including bond dimensions and aggregation."""

    enc_bond_input_dim: int
    enc_bond_hidden_dim: int
    enc_aggregation: str
    enc_jk: str


class GatedGCNMixin(BaseDataModel):
    """Mixin schema for GatedGCN-specific parameters."""

    enc_norm: str | None
    enc_bond_input_dim: int
    enc_jk: str


class E3GNNMixin(BaseDataModel):
    """Mixin schema for E(3)-equivariant GNN parameters including coordinate updates."""

    enc_bond_input_dim: int
    enc_m_dim: int
    enc_fourier_features: int
    enc_soft_edge: bool
    enc_norm_feats: bool
    enc_norm_coors: bool
    enc_update_coors: bool
    enc_jk: str


class GPSMixin(BaseDataModel):
    """Mixin schema for GPS (General Powerful Scalable) graph transformer parameters."""

    enc_bond_input_dim: int
    enc_num_heads: int
    enc_expansion_k: int
    enc_distance_k: int | None
    enc_norm: str | None


class GPS3DMixin(BaseDataModel):
    """Mixin schema for 3D-aware GPS graph transformer parameters."""

    enc_bond_input_dim: int
    enc_num_heads: int
    enc_expansion_k: int
    enc_num_kernels: int
    enc_norm: str | None
    enc_jk: str


class GTMixin(BaseDataModel):
    """Mixin schema for Graph Transformer encoder parameters."""

    enc_bond_input_dim: int
    enc_num_heads: int
    enc_expansion_k: int
    enc_distance_k: int | None
    enc_jk: str


class GT3DMixin(BaseDataModel):
    """Mixin schema for 3D-aware Graph Transformer encoder parameters."""

    enc_bond_input_dim: int
    enc_num_heads: int
    enc_expansion_k: int
    enc_num_kernels: int
    enc_jk: str


# ------------------------------------------------------------------------------
#                              CLMs
# ------------------------------------------------------------------------------


class CLMMixin(BaseDataModel):
    """Mixin schema for character-level model (CLM) shared parameters."""

    enc_num_characters: int
    pred_hidden_dims: list[int] | None


class CNNMixin(BaseDataModel):
    """Mixin schema for CNN encoder parameters including kernel dimensions."""

    enc_hidden_dim: int
    enc_kernel_dims: list[int]
    enc_num_heads: int


class RNNMixin(BaseDataModel):
    """Mixin schema for RNN encoder parameters including type and bidirectionality."""

    enc_rnn_type: str
    enc_hidden_dim: int
    enc_bidirectional: bool
    enc_num_layers: int
    enc_embedding_dim: int
    enc_num_heads: int


class RoFormerMixin(BaseDataModel):
    """Mixin schema for RoFormer encoder parameters with rotary position embeddings."""

    enc_hidden_dim: int
    enc_expansion_dim: int
    enc_num_heads: int
    enc_num_layers: int
    enc_attention_dropout: float
    enc_hidden_dropout: float


# ------------------------------------------------------------------------------
#                               DNNs
# ------------------------------------------------------------------------------


class TabularMixin(BaseDataModel):
    """Mixin schema for tabular (deep lasso) model parameters."""

    deep_lasso_weight: float
    dropout: float


class MLPMixin(BaseDataModel):
    """Mixin schema for MLP predictor parameters including hidden dimensions."""

    hidden_dims: list[int] | None
    activation: str
    task_head_dims: list[int] | None


class SNNMixin(BaseDataModel):
    """Mixin schema for Self-Normalizing Network parameters."""

    hidden_dims: list[int] | None
    num_parallel: int


# ------------------------------------------------------------------------------
#                               Finetuner
# ------------------------------------------------------------------------------


class FinetunerMixin(BaseDataModel):
    """Mixin schema for finetuning a pretrained encoder with LoRA or full strategy."""

    architecture: str
    path_to_pretrained: str
    pred_hidden_dims: list[int] | None
    activation: str
    dropout: float
    num_endpoints: int
    loss_fn: str
    loss_args: dict
    optimizer: str
    optimizer_args: dict
    pretrain_lr: float
    pretrain_decay: float
    scheduler: str
    scheduler_args: dict
    finetuning_strategy: str = "full"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_min_dim: int = 32


class ChempropFinetunerMixin(BaseDataModel):
    """Mixin schema for finetuning a Chemprop pretrained model."""

    path_to_pretrained: str
    num_endpoints: int
    loss_fn: str
    optimizer: str
    optimizer_args: dict
    scheduler_args: dict
    pred_hidden_dim: int | None
    pred_num_layers: int
    pred_dropout: float
    pred_activation: str
