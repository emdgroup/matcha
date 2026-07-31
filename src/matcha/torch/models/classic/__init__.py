"""Classic (single-task/multitask) model wrappers combining an encoder with a prediction head."""

from matcha.torch.models.classic.attentivefp_model import AttentiveFPModel
from matcha.torch.models.classic.base_classic_model import BaseClassicModel
from matcha.torch.models.classic.roformer_model import RoFormerModel
from matcha.torch.models.classic.chemprop_model import ChempropModel
from matcha.torch.models.classic.cnn_model import CNNModel
from matcha.torch.models.classic.e3gnn_model import E3GNNModel
from matcha.torch.models.classic.gatedcgn_model import GatedGCNModel
from matcha.torch.models.classic.gin_model import GINModel
from matcha.torch.models.classic.gps3d_model import GPS3DModel
from matcha.torch.models.classic.gps_model import GPSModel
from matcha.torch.models.classic.gt_model import GTModel
from matcha.torch.models.classic.gt3d_model import GT3DModel
from matcha.torch.models.classic.mlp_model import MLPModel
from matcha.torch.models.classic.snn_model import SNNModel
from matcha.torch.models.classic.rnn_model import RNNModel

__all__ = [
    "GINModel",
    "AttentiveFPModel",
    "GatedGCNModel",
    "CNNModel",
    "RNNModel",
    "MLPModel",
    "SNNModel",
    "E3GNNModel",
    "BaseClassicModel",
    "RoFormerModel",
    "GPSModel",
    "GPS3DModel",
    "ChempropModel",
    "GTModel",
    "GT3DModel",
]
