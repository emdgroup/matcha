"""Pretraining models for self-supervised learning.

This module contains model variants adapted for self-supervised learning tasks:
- RoFormerMLM: RoFormer variant for Masked Language Modeling (MLM)
- Graph pretraining models: GIN, GatedGCN, GPS, GT, AttentiveFP variants that
  support both node-level and graph-level predictions
"""

from matcha.torch.models.pretraining.base_pretraining_model import (
    BasePretrainingModel,
    PretrainingModelRegistry,
)
from matcha.torch.models.pretraining.base_graph_pretraining import (
    BaseGraphPretrainingModel,
)
from matcha.torch.models.pretraining.roformer_mlm import RoFormerMLM
from matcha.torch.models.pretraining.gin_pretraining import GINPretraining
from matcha.torch.models.pretraining.gatedgcn_pretraining import GatedGCNPretraining
from matcha.torch.models.pretraining.gps_pretraining import GPSPretraining
from matcha.torch.models.pretraining.gt_pretraining import GTPretraining
from matcha.torch.models.pretraining.attentivefp_pretraining import (
    AttentiveFPPretraining,
)
from matcha.torch.models.pretraining.e3gnn_pretraining import E3GNNPretraining

__all__ = [
    "BasePretrainingModel",
    "BaseGraphPretrainingModel",
    "PretrainingModelRegistry",
    "RoFormerMLM",
    "GINPretraining",
    "GatedGCNPretraining",
    "GPSPretraining",
    "GTPretraining",
    "AttentiveFPPretraining",
    "E3GNNPretraining",
]
