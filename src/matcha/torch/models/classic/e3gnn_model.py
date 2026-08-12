"""E(3)-equivariant graph neural network (E3GNN) classic model."""

from matcha.torch.models.classic.base_classic_model import (
    BaseClassicModel,
    ClassicModelRegistry,
)
from matcha.torch.encoders.e3gnn import E3GNN
from matcha.utils.schemas import E3GNNInputModel

from lightning.pytorch.core.mixins import HyperparametersMixin


@ClassicModelRegistry.register()
class E3GNNModel(BaseClassicModel, HyperparametersMixin):
    """E(3)-equivariant graph neural network (E3GNN) for molecular property
    prediction from 3-D conformers.

    Uses E(n)-equivariant message passing that operates on both node features
    and 3-D coordinates. Inherits from :class:`BaseClassicModel` for common
    training/prediction routines and from
    :class:`~lightning.pytorch.core.mixins.HyperparametersMixin` for saving
    hyperparameters.

    Reference: Satorras et al., *E(n) Equivariant Graph Neural Networks*
    (https://arxiv.org/abs/2102.09844)

    Example usage:

    .. code-block:: python

        model = E3GNNModel()
        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param int additional_mol_features_dim: dimensionality of extra molecular
        features concatenated to encoder output, defaults to 0
    :param int enc_num_layers: number of equivariant message-passing layers,
        defaults to 3
    :param int enc_atom_input_dim: input atom feature dimensionality, defaults to 44
    :param int enc_bond_input_dim: input bond feature dimensionality, defaults to 14
    :param int enc_atom_hidden_dim: hidden atom feature dimensionality, defaults to 128
    :param int enc_m_dim: message dimensionality, defaults to 16
    :param int enc_fourier_features: number of Fourier features for distance encoding,
        defaults to 4
    :param bool enc_soft_edge: whether to use soft edge weights, defaults to False
    :param bool enc_norm_feats: whether to normalise node features, defaults to True
    :param bool enc_norm_coors: whether to normalise coordinate updates, defaults to True
    :param bool enc_update_coors: whether to update coordinates, defaults to True
    :param float enc_coor_weights_clamp_value: symmetric clamp on per-edge
        coordinate weights (paper's ``torch.clamp(min=-100, max=100)``),
        defaults to 100.0
    :param float enc_norm_coors_scale_init: initial value of the learnable
        scale parameter inside :class:`CoorsNorm` when ``enc_norm_coors=True``,
        defaults to 1e-2
    :param str enc_jk: jumping knowledge strategy, defaults to 'last'
    :param str enc_readout: graph-level readout strategy, defaults to 'vpa'
    :param str enc_activation: activation function in the encoder, defaults to 'swish'
    :param float enc_dropout: dropout rate in the encoder, defaults to 0.2
    :param int enc_laplacian_k: Laplacian positional encoding dimension, defaults to 10
    :param int enc_rwse_k: random-walk structural encoding dimension, defaults to 20
    :param int enc_elstatic_k: electrostatic encoding dimension, defaults to 0
    :param int enc_distmat_k: distance matrix encoding dimension, defaults to 0
    :param int enc_rrwp_k: relative random-walk probabilities dimension, defaults to 0
    :param list[int] pred_hidden_dims: hidden layer sizes in the MLP predictor,
        defaults to [512, 256]
    :param list[int] | None pred_task_head_dims: per-task head dimensions, defaults to None
    :param str pred_activation: activation in the predictor, defaults to 'swish'
    :param float pred_dropout: dropout rate in the predictor, defaults to 0.2
    :param int num_endpoints: number of prediction targets, defaults to 1
    :param str loss_fn: loss function name, defaults to 'mse'
    :param dict loss_args: additional loss function arguments, defaults to {}
    :param str optimizer: optimizer name, defaults to 'adam'
    :param dict optimizer_args: additional optimizer arguments, defaults to {'lr': 1e-3}
    :param str scheduler: learning rate scheduler name, defaults to 'cosine_annealing'
    :param dict scheduler_args: additional scheduler arguments,
        defaults to {'min_lr': 1e-6, 'total_steps': 50}
    """

    def __init__(
        self,
        additional_mol_features_dim: int = 0,
        enc_num_layers: int = 3,
        enc_atom_input_dim: int = 44,
        enc_bond_input_dim: int = 14,
        enc_atom_hidden_dim: int = 128,
        enc_m_dim: int = 16,
        enc_fourier_features: int = 4,
        enc_soft_edge: bool = False,
        enc_norm_feats: bool = True,
        enc_norm_coors: bool = True,
        enc_update_coors: bool = True,
        enc_coor_weights_clamp_value: float = 100.0,
        enc_norm_coors_scale_init: float = 1e-2,
        enc_jk: str = "last",
        enc_readout: str = "vpa",
        enc_activation: str = "swish",
        enc_dropout: float = 0.2,
        enc_laplacian_k: int = 10,
        enc_rwse_k: int = 20,
        enc_elstatic_k: int = 0,
        enc_distmat_k: int = 0,
        enc_rrwp_k: int = 0,
        pred_hidden_dims: list[int] = [512, 256],
        pred_task_head_dims: list[int] | None = None,
        pred_activation: str = "swish",
        pred_dropout: float = 0.2,
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 1e-3},
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {"min_lr": 1e-6, "total_steps": 50},
    ):
        super().__init__(additional_mol_features_dim=additional_mol_features_dim)
        self.save_hyperparameters()
        self.params = E3GNNInputModel(**self.hparams)
        atom_input_dim = (
            enc_atom_input_dim
            + enc_laplacian_k
            + enc_rwse_k
            + enc_distmat_k
            + enc_elstatic_k
        )
        edge_input_dim = enc_bond_input_dim + enc_rrwp_k
        self.encoder = E3GNN(
            num_layers=enc_num_layers,
            atom_input_dim=atom_input_dim,
            bond_input_dim=edge_input_dim,
            atom_hidden_dim=enc_atom_hidden_dim,
            m_dim=enc_m_dim,
            fourier_features=enc_fourier_features,
            soft_edge=enc_soft_edge,
            norm_feats=enc_norm_feats,
            norm_coors=enc_norm_coors,
            update_coors=enc_update_coors,
            activation=enc_activation,
            dropout=enc_dropout,
            coor_weights_clamp_value=enc_coor_weights_clamp_value,
            norm_coors_scale_init=enc_norm_coors_scale_init,
            jk=enc_jk,
            readout=enc_readout,
            laplacian_k=enc_laplacian_k,
            rwse_k=enc_rwse_k,
            elstatic_k=enc_elstatic_k,
            distmat_k=enc_distmat_k,
            rrwp_k=enc_rrwp_k,
        )
        self._parse_predictor()
        self._parse_train_config()
