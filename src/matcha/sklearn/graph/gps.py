"""Sklearn-compatible GPS (General, Powerful, Scalable Graph Transformer) models."""

from matcha.torch.models.classic import GPSModel
from matcha.sklearn.base_sklearn_model import (
    ScikitLearnModelRegistry,
    ScikitLearnClassifierMixin,
    ScikitLearnRegressorMixin,
)
from matcha.sklearn.graph.base_sklearn_gnn import BaseScikitLearnGNN


@ScikitLearnModelRegistry.register()
class GPSClassifier(BaseScikitLearnGNN, ScikitLearnClassifierMixin):
    """General, Powerful, Scalable Graph Transformer (GPS) classifier.

    Combines local message-passing with global self-attention in a hybrid
    architecture for molecular property prediction. Compatible with
    classification datasets only.

    Inherits from :class:`~matcha.sklearn.graph.BaseScikitLearnGNN` for
    graph-specific datamodule creation and collate-aware dataloading.

    References:

    - Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*
      https://arxiv.org/abs/2205.12454

    Example usage:

    .. code-block:: python

        model = GPSClassifier()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_num_layers: number of encoder layers, defaults to 3
    :param int enc_atom_hidden_dim: encoder output atom feature dimensionality,
        defaults to 256
    :param str enc_jk: encoder jumping knowledge strategy, defaults to 'last'
    :param str | None enc_norm: normalization layer type, defaults to 'layer'
    :param int enc_num_heads: number of attention heads, defaults to 8
    :param float enc_expansion_k: FFN expansion factor, defaults to 2
    :param int | None enc_distance_k: shortest-path distance clipping, defaults to 10
    :param str enc_readout: encoder aggregation for molecule-level embeddings,
        defaults to 'vpa'
    :param float enc_dropout: dropout rate across the encoder, defaults to 0.2
    :param str enc_activation: activation function in the encoder, defaults to 'gelu'
    :param list[int] | None pred_hidden_dims: shape of hidden MLP layers in the
        predictor, defaults to [256, 256]
    :param list[int] | None pred_task_head_dims: shape of per-task MLP layers,
        defaults to None
    :param str pred_activation: activation function in the predictor, defaults
        to 'gelu'
    :param float pred_dropout: dropout rate in the predictor, defaults to 0.2
    :param int num_endpoints: number of endpoints or classes to predict, defaults to 1
    :param str loss_fn: loss function to optimize, defaults to 'bce'
    :param dict loss_args: additional arguments for the loss function, defaults to {}
    :param str optimizer: optimizer to use, defaults to 'adamw'
    :param dict optimizer_args: optimizer arguments, defaults to
        {'lr': 1e-4, 'weight_decay': 1e-4}
    :param str scheduler: learning rate scheduler, defaults to 'warmup_linear_decay'
    :param dict scheduler_args: scheduler arguments, defaults to {'min_lr': 1e-5}
    :param int num_epochs: number of training epochs, defaults to 100
    :param int batch_size: batch size for training and prediction, defaults to 64
    :param bool stochastic_weight_averaging: whether to use SWA, defaults to False
    :param bool early_stopping: whether to use early stopping, defaults to True
    :param int patience: epochs to wait before early stopping, defaults to 20
    :param int devices: number of devices for training, defaults to 1
    :param str accelerator: hardware accelerator, defaults to 'gpu'
    :param int rwse_k: random walk structural encoding dimensions, defaults to 20
    :param int laplacian_k: Laplacian PE components, defaults to 0
    :param int elstatic_k: electrostatic encoding dimensions, defaults to 0
    :param int distmat_k: distance matrix encoding dimensions, defaults to 0
    :param int rrwp_k: relative random walk probability dimensions, defaults to 20
    :param int num_virtual_nodes: number of virtual nodes, defaults to 0
    :param list[str] | None feature_list: molecular feature set to compute,
        defaults to None
    :param dict label_encoder_params: label encoder parameters, defaults to {}
    :param str | list[str] | dict | None label_transform_map: label transform
        specification, defaults to None
    :param bool augment_resonance: whether to augment with resonance structures,
        defaults to False
    :param int seed: random seed, defaults to 0
    """

    def __init__(
        self,
        enc_num_layers: int = 3,
        enc_atom_hidden_dim: int = 256,
        enc_jk: str = "last",
        enc_norm: str | None = "layer",
        enc_num_heads: int = 8,
        enc_expansion_k: float = 2,
        enc_distance_k: int | None = 10,
        enc_readout: str = "vpa",
        enc_dropout: float = 0.2,
        enc_activation: str = "gelu",
        pred_hidden_dims: list[int] | None = [256, 256],
        pred_task_head_dims: list[int] | None = None,
        pred_activation: str = "gelu",
        pred_dropout: float = 0.2,
        num_endpoints: int = 1,
        loss_fn: str = "bce",
        loss_args: dict = {},
        optimizer: str = "adamw",
        optimizer_args: dict = {"lr": 1e-4, "weight_decay": 1e-4},
        scheduler: str = "warmup_linear_decay",
        scheduler_args: dict = {"min_lr": 1e-5},
        num_epochs: int = 100,
        batch_size: int = 64,
        stochastic_weight_averaging: bool = False,
        early_stopping: bool = True,
        patience: int = 20,
        devices: int = 1,
        accelerator: str = "gpu",
        rwse_k: int = 20,
        laplacian_k: int = 0,
        elstatic_k: int = 0,
        distmat_k: int = 0,
        rrwp_k: int = 20,
        num_virtual_nodes: int = 0,
        feature_list: list[str] | None = None,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        augment_resonance: bool = False,
        seed: int = 0,
    ):
        self._architecture = GPSModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(GPSClassifier, self).__init__(params)


@ScikitLearnModelRegistry.register()
class GPSRegressor(BaseScikitLearnGNN, ScikitLearnRegressorMixin):
    """General, Powerful, Scalable Graph Transformer (GPS) regressor.

    Combines local message-passing with global self-attention in a hybrid
    architecture for molecular property prediction. Compatible with
    regression datasets only.

    Inherits from :class:`~matcha.sklearn.graph.BaseScikitLearnGNN` for
    graph-specific datamodule creation and collate-aware dataloading.

    References:

    - Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*
      https://arxiv.org/abs/2205.12454

    Example usage:

    .. code-block:: python

        model = GPSRegressor()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_num_layers: number of encoder layers, defaults to 3
    :param int enc_atom_hidden_dim: encoder output atom feature dimensionality,
        defaults to 256
    :param str enc_jk: encoder jumping knowledge strategy, defaults to 'last'
    :param str | None enc_norm: normalization layer type, defaults to 'layer'
    :param int enc_num_heads: number of attention heads, defaults to 8
    :param float enc_expansion_k: FFN expansion factor, defaults to 2
    :param int | None enc_distance_k: shortest-path distance clipping, defaults to 10
    :param str enc_readout: encoder aggregation for molecule-level embeddings,
        defaults to 'vpa'
    :param float enc_dropout: dropout rate across the encoder, defaults to 0.2
    :param str enc_activation: activation function in the encoder, defaults to 'gelu'
    :param list[int] | None pred_hidden_dims: shape of hidden MLP layers in the
        predictor, defaults to [256, 256]
    :param list[int] | None pred_task_head_dims: shape of per-task MLP layers,
        defaults to None
    :param str pred_activation: activation function in the predictor, defaults
        to 'gelu'
    :param float pred_dropout: dropout rate in the predictor, defaults to 0.2
    :param int num_endpoints: number of endpoints to predict, defaults to 1
    :param str loss_fn: loss function to optimize, defaults to 'mse'
    :param dict loss_args: additional arguments for the loss function, defaults to {}
    :param str optimizer: optimizer to use, defaults to 'adamw'
    :param dict optimizer_args: optimizer arguments, defaults to
        {'lr': 1e-4, 'weight_decay': 1e-4}
    :param str scheduler: learning rate scheduler, defaults to 'warmup_linear_decay'
    :param dict scheduler_args: scheduler arguments, defaults to {'min_lr': 1e-5}
    :param int num_epochs: number of training epochs, defaults to 100
    :param int batch_size: batch size for training and prediction, defaults to 64
    :param bool stochastic_weight_averaging: whether to use SWA, defaults to False
    :param bool early_stopping: whether to use early stopping, defaults to True
    :param int patience: epochs to wait before early stopping, defaults to 20
    :param int devices: number of devices for training, defaults to 1
    :param str accelerator: hardware accelerator, defaults to 'gpu'
    :param int rwse_k: random walk structural encoding dimensions, defaults to 20
    :param int laplacian_k: Laplacian PE components, defaults to 0
    :param int elstatic_k: electrostatic encoding dimensions, defaults to 0
    :param int distmat_k: distance matrix encoding dimensions, defaults to 0
    :param int rrwp_k: relative random walk probability dimensions, defaults to 20
    :param int num_virtual_nodes: number of virtual nodes, defaults to 0
    :param list[str] | None feature_list: molecular feature set to compute,
        defaults to None
    :param bool clip: whether to clip predictions to training label range,
        defaults to True
    :param dict label_encoder_params: label encoder parameters, defaults to {}
    :param str | list[str] | dict | None label_transform_map: label transform
        specification, defaults to None
    :param str scaler_type: type of target scaler, defaults to 'standard'
    :param bool augment_resonance: whether to augment with resonance structures,
        defaults to False
    :param int seed: random seed, defaults to 0
    """

    def __init__(
        self,
        enc_num_layers: int = 3,
        enc_atom_hidden_dim: int = 256,
        enc_jk: str = "last",
        enc_norm: str | None = "layer",
        enc_num_heads: int = 8,
        enc_expansion_k: float = 2,
        enc_distance_k: int | None = 10,
        enc_readout: str = "vpa",
        enc_dropout: float = 0.2,
        enc_activation: str = "gelu",
        pred_hidden_dims: list[int] | None = [256, 256],
        pred_task_head_dims: list[int] | None = None,
        pred_activation: str = "gelu",
        pred_dropout: float = 0.2,
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adamw",
        optimizer_args: dict = {"lr": 1e-4, "weight_decay": 1e-4},
        scheduler: str = "warmup_linear_decay",
        scheduler_args: dict = {"min_lr": 1e-5},
        num_epochs: int = 100,
        batch_size: int = 64,
        stochastic_weight_averaging: bool = False,
        early_stopping: bool = True,
        patience: int = 20,
        devices: int = 1,
        accelerator: str = "gpu",
        rwse_k: int = 20,
        laplacian_k: int = 0,
        elstatic_k: int = 0,
        distmat_k: int = 0,
        rrwp_k: int = 20,
        num_virtual_nodes: int = 0,
        feature_list: list[str] | None = None,
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        scaler_type: str = "standard",
        augment_resonance: bool = False,
        seed: int = 0,
    ):
        self._architecture = GPSModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(GPSRegressor, self).__init__(params)
