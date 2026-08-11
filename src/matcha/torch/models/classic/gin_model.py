"""Graph Isomorphism Network (GIN) classic model."""

from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.torch.encoders.gin import GIN
from matcha.torch.models.classic.base_classic_model import (
    BaseClassicModel,
    ClassicModelRegistry,
)
from matcha.utils.schemas import GINInputModel


@ClassicModelRegistry.register()
class GINModel(BaseClassicModel, HyperparametersMixin):
    """Graph Isomorphism Network (GIN) for molecular property prediction.

    Learns to embed molecular graphs via message-passing GIN layers and predict
    properties in an end-to-end manner. Inherits from :class:`BaseClassicModel`
    for common training/prediction routines and from
    :class:`~lightning.pytorch.core.mixins.HyperparametersMixin` for saving
    hyperparameters.

    Reference: Xu et al., *How Powerful are Graph Neural Networks?*
    (https://arxiv.org/abs/1810.00826)

    Example usage:

    .. code-block:: python

        model = GINModel()
        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param int additional_mol_features_dim: dimensionality of extra molecular
        features concatenated to encoder output, defaults to 0
    :param int enc_num_layers: number of GIN message-passing layers, defaults to 6
    :param int enc_atom_input_dim: input atom feature dimensionality, defaults to 44
    :param int enc_bond_input_dim: input bond feature dimensionality, defaults to 14
    :param int enc_atom_hidden_dim: hidden atom feature dimensionality, defaults to 300
    :param str enc_aggregation: neighbour aggregation strategy, defaults to 'vpa'
    :param str enc_jk: jumping knowledge strategy, defaults to 'concat'
    :param str | None enc_norm: normalisation inside GIN layers, defaults to 'graph'
    :param str enc_readout: graph-level readout strategy, defaults to 'vpa'
    :param str enc_activation: activation function in the encoder, defaults to 'swish'
    :param float enc_dropout: dropout rate in the encoder, defaults to 0.2
    :param int enc_laplacian_k: Laplacian positional encoding dimension, defaults to 10
    :param int enc_rwse_k: random-walk structural encoding dimension, defaults to 20
    :param int enc_elstatic_k: electrostatic encoding dimension, defaults to 0
    :param int enc_distmat_k: distance matrix encoding dimension, defaults to 0
    :param int enc_rrwp_k: relative random-walk probabilities dimension, defaults to 20
    :param float enc_eps: initial value of the ``eps`` term in ``GINEConv``, defaults to 0.0
    :param bool enc_train_eps: whether to learn ``eps`` as a parameter (GIN-ε variant),
        defaults to True
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
        enc_num_layers: int = 6,
        enc_atom_input_dim: int = 44,
        enc_bond_input_dim: int = 14,
        enc_atom_hidden_dim: int = 300,
        enc_aggregation: str = "vpa",
        enc_jk: str = "concat",
        enc_norm: str | None = "graph",
        enc_readout: str = "vpa",
        enc_activation: str = "swish",
        enc_dropout: float = 0.2,
        enc_laplacian_k: int = 10,
        enc_rwse_k: int = 20,
        enc_elstatic_k: int = 0,
        enc_distmat_k: int = 0,
        enc_rrwp_k: int = 20,
        enc_eps: float = 0.0,
        enc_train_eps: bool = True,
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

        self.params = GINInputModel(**self.hparams)
        atom_input_dim = (
            enc_atom_input_dim
            + enc_laplacian_k
            + enc_rwse_k
            + enc_distmat_k
            + enc_elstatic_k
        )
        edge_input_dim = enc_bond_input_dim + enc_rrwp_k

        self.encoder = GIN(
            num_layers=enc_num_layers,
            atom_input_dim=atom_input_dim,
            bond_input_dim=edge_input_dim,
            atom_hidden_dim=enc_atom_hidden_dim,
            activation=enc_activation,
            aggregation=enc_aggregation,
            dropout=enc_dropout,
            norm=enc_norm,
            jk=enc_jk,
            readout=enc_readout,
            laplacian_k=enc_laplacian_k,
            rwse_k=enc_rwse_k,
            elstatic_k=enc_elstatic_k,
            distmat_k=enc_distmat_k,
            rrwp_k=enc_rrwp_k,
            eps=enc_eps,
            train_eps=enc_train_eps,
        )
        self._parse_predictor()
        self._parse_train_config()
