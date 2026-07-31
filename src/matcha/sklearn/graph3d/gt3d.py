"""Graph Transformer 3D models for molecular conformers."""

from matcha.torch.models.classic import GT3DModel
from matcha.sklearn.base_sklearn_model import (
    ScikitLearnModelRegistry,
    ScikitLearnClassifierMixin,
    ScikitLearnRegressorMixin,
)
from matcha.sklearn.graph3d.base_sklearn_gnn3d import BaseScikitLearnGNN3D


@ScikitLearnModelRegistry.register()
class GT3DClassifier(BaseScikitLearnGNN3D, ScikitLearnClassifierMixin):
    """Graph Transformer 3D classifier for molecular conformers.

    3D extension of the Graph Transformer architecture loosely inspired by the
    `gt-pyg <https://github.com/pyg-team/pytorch_geometric>`_ implementation.
    Uses multi-head self-attention over 3D molecular conformers with positional
    encodings derived from the molecular geometry. Compatible with classification
    datasets only.

    Inherits from :class:`~matcha.sklearn.graph3d.BaseScikitLearnGNN3D`
    for common training and predicting routines.

    Example usage:

    .. code-block:: python

        model = GT3DClassifier()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_atom_hidden_dim: Atom feature dimensionality,
        defaults to 256.
    :param int enc_num_layers: Number of transformer layers, defaults to 3.
    :param str enc_jk: Jumping knowledge strategy, defaults to 'last'.
    :param int enc_num_heads: Number of attention heads, defaults to 8.
    :param int enc_expansion_k: FFN expansion factor, defaults to 2.
    :param int enc_num_kernels: Number of convolution kernels, defaults to 128.
    :param str enc_readout: Aggregation strategy for molecule-level embeddings,
        defaults to 'vpa'.
    :param float enc_dropout: Encoder dropout rate, defaults to 0.2.
    :param str enc_activation: Encoder activation function, defaults to 'gelu'.
    :param list[int] | None pred_hidden_dims: Hidden layer dimensions in the
        predictor MLP, defaults to [256, 256].
    :param list[int] | None pred_task_head_dims: Per-task head dimensions,
        defaults to None.
    :param str pred_activation: Predictor activation function, defaults to 'gelu'.
    :param float pred_dropout: Predictor dropout rate, defaults to 0.2.
    :param int num_endpoints: Number of endpoints or classes to predict,
        defaults to 1.
    :param str loss_fn: Loss function, defaults to 'bce'.
    :param dict loss_args: Additional loss function arguments, defaults to {}.
    :param str optimizer: Optimizer, defaults to 'adamw'.
    :param dict optimizer_args: Optimizer arguments, defaults to
        {'lr': 1e-4, 'weight_decay': 1e-4}.
    :param str scheduler: Learning rate scheduler, defaults to
        'warmup_linear_decay'.
    :param dict scheduler_args: Scheduler arguments, defaults to
        {'min_lr': 1e-5}. If 'total_steps' is not provided, it is
        auto-computed as num_epochs * ceil(len(train_data) / batch_size).
    :param int num_epochs: Number of training epochs, defaults to 100.
    :param int batch_size: Batch size for training and prediction, defaults to 64.
    :param bool stochastic_weight_averaging: Whether to apply SWA after regular
        training, defaults to False.
    :param bool early_stopping: Whether to enable early stopping, defaults to True.
    :param int patience: Early stopping patience in epochs, defaults to 20.
    :param int devices: Number of devices for training, defaults to 1.
    :param str accelerator: Accelerator type ('cpu', 'gpu', 'tpu', 'hpu'),
        defaults to 'gpu'.
    :param int rwse_k: Random walk structural encoding steps, defaults to 20.
    :param int laplacian_k: Laplacian positional encoding components,
        defaults to 0.
    :param int elstatic_k: Electrostatic positional encoding components,
        defaults to 0.
    :param int distmat_k: Distance matrix positional encoding components,
        defaults to 0.
    :param int rrwp_k: Relative random walk positional encoding steps,
        defaults to 20.
    :param int num_virtual_nodes: Number of virtual nodes to add, defaults to 0.
    :param list[str] | None feature_list: Tabular features to combine with
        graph features, defaults to None.
    :param dict label_encoder_params: Label encoder parameters, defaults to {}.
    :param str | list[str] | dict | None label_transform_map: Label transform
        configuration, defaults to None.
    :param bool augment_resonance: Whether to augment with resonance structures,
        defaults to False.
    :param int seed: Random seed, defaults to 0.
    """

    def __init__(
        self,
        enc_atom_hidden_dim: int = 256,
        enc_num_layers: int = 3,
        enc_jk: str = "last",
        enc_num_heads: int = 8,
        enc_expansion_k: int = 2,
        enc_num_kernels: int = 128,
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
        self._architecture = GT3DModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(GT3DClassifier, self).__init__(params)


@ScikitLearnModelRegistry.register()
class GT3DRegressor(BaseScikitLearnGNN3D, ScikitLearnRegressorMixin):
    """Graph Transformer 3D regressor for molecular conformers.

    3D extension of the Graph Transformer architecture loosely inspired by the
    `gt-pyg <https://github.com/pyg-team/pytorch_geometric>`_ implementation.
    Uses multi-head self-attention over 3D molecular conformers with positional
    encodings derived from the molecular geometry. Compatible with regression
    datasets only.

    Inherits from :class:`~matcha.sklearn.graph3d.BaseScikitLearnGNN3D`
    for common training and predicting routines.

    Example usage:

    .. code-block:: python

        model = GT3DRegressor()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_atom_hidden_dim: Atom feature dimensionality,
        defaults to 256.
    :param int enc_num_layers: Number of transformer layers, defaults to 3.
    :param str enc_jk: Jumping knowledge strategy, defaults to 'last'.
    :param int enc_num_heads: Number of attention heads, defaults to 8.
    :param int enc_expansion_k: FFN expansion factor, defaults to 2.
    :param int enc_num_kernels: Number of convolution kernels, defaults to 128.
    :param str enc_readout: Aggregation strategy for molecule-level embeddings,
        defaults to 'vpa'.
    :param float enc_dropout: Encoder dropout rate, defaults to 0.2.
    :param str enc_activation: Encoder activation function, defaults to 'gelu'.
    :param list[int] | None pred_hidden_dims: Hidden layer dimensions in the
        predictor MLP, defaults to [256, 256].
    :param list[int] | None pred_task_head_dims: Per-task head dimensions,
        defaults to None.
    :param str pred_activation: Predictor activation function, defaults to 'gelu'.
    :param float pred_dropout: Predictor dropout rate, defaults to 0.2.
    :param int num_endpoints: Number of endpoints to predict, defaults to 1.
    :param str loss_fn: Loss function, defaults to 'mse'.
    :param dict loss_args: Additional loss function arguments, defaults to {}.
    :param str optimizer: Optimizer, defaults to 'adamw'.
    :param dict optimizer_args: Optimizer arguments, defaults to
        {'lr': 1e-4, 'weight_decay': 1e-4}.
    :param str scheduler: Learning rate scheduler, defaults to
        'warmup_linear_decay'.
    :param dict scheduler_args: Scheduler arguments, defaults to
        {'min_lr': 1e-5}. If 'total_steps' is not provided, it is
        auto-computed as num_epochs * ceil(len(train_data) / batch_size).
    :param int num_epochs: Number of training epochs, defaults to 100.
    :param int batch_size: Batch size for training and prediction, defaults to 64.
    :param bool stochastic_weight_averaging: Whether to apply SWA after regular
        training, defaults to False.
    :param bool early_stopping: Whether to enable early stopping, defaults to True.
    :param int patience: Early stopping patience in epochs, defaults to 20.
    :param int devices: Number of devices for training, defaults to 1.
    :param str accelerator: Accelerator type ('cpu', 'gpu', 'tpu', 'hpu'),
        defaults to 'gpu'.
    :param int rwse_k: Random walk structural encoding steps, defaults to 20.
    :param int laplacian_k: Laplacian positional encoding components,
        defaults to 0.
    :param int elstatic_k: Electrostatic positional encoding components,
        defaults to 0.
    :param int distmat_k: Distance matrix positional encoding components,
        defaults to 0.
    :param int rrwp_k: Relative random walk positional encoding steps,
        defaults to 20.
    :param int num_virtual_nodes: Number of virtual nodes to add, defaults to 0.
    :param list[str] | None feature_list: Tabular features to combine with
        graph features, defaults to None.
    :param bool clip: Whether to clip predictions to the training label range,
        defaults to True.
    :param dict label_encoder_params: Label encoder parameters, defaults to {}.
    :param str | list[str] | dict | None label_transform_map: Label transform
        configuration, defaults to None.
    :param str scaler_type: Type of target scaler, defaults to 'standard'.
    :param bool augment_resonance: Whether to augment with resonance structures,
        defaults to False.
    :param int seed: Random seed, defaults to 0.
    """

    def __init__(
        self,
        enc_atom_hidden_dim: int = 256,
        enc_num_layers: int = 3,
        enc_jk: str = "last",
        enc_num_heads: int = 8,
        enc_expansion_k: int = 2,
        enc_num_kernels: int = 128,
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
        self._architecture = GT3DModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(GT3DRegressor, self).__init__(params)
