"""Sklearn-compatible 1D CNN wrappers for molecular property prediction from chemical language."""

from matcha.sklearn.clm.base_sklearn_clm import BaseScikitLearnCLM
from matcha.torch.models.classic import CNNModel
from matcha.sklearn.base_sklearn_model import (
    ScikitLearnModelRegistry,
    ScikitLearnClassifierMixin,
    ScikitLearnRegressorMixin,
)


@ScikitLearnModelRegistry.register()
class CNNClassifier(BaseScikitLearnCLM, ScikitLearnClassifierMixin):
    """1D convolutional neural network (CNN) for molecular property classification.

    Learns to embed chemical language (e.g. SMILES) via convolutional layers
    and predict properties in an end-to-end manner. Only compatible with
    classification datasets.
    Inherits from :class:`~matcha.sklearn.clm.base_sklearn_clm.BaseScikitLearnCLM`.

    References:

    - Boldini et al., *J. Cheminform.* (2024): https://arxiv.org/abs/2407.12152
    - Boldini et al., *Digital Discovery* (2023):
      https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g

    Example usage:

    .. code-block:: python

        model = CNNClassifier()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_hidden_dim: number of filters per convolutional layer,
        defaults to 256

    :param list[int] enc_kernel_dims: kernel sizes for each convolutional layer,
        defaults to [3, 5, 8, 11, 14]

    :param int enc_num_heads: number of attention heads for computing self-attention,
        defaults to 8

    :param str enc_activation: activation function to use throughout the encoder,
        defaults to 'swish'

    :param float enc_dropout: dropout rate across the encoder, defaults to 0.2

    :param list[int] pred_hidden_dims: shape of hidden MLP layers in the predictor,
        defaults to [256, 256]

    :param list[int] | None pred_task_head_dims: shape of the MLP layers
        dedicated to each task independently, defaults to None

    :param str pred_activation: activation function to use in the predictor, defaults
        to 'swish'

    :param float pred_dropout: dropout rate across the predictor, defaults to 0.2

    :param int num_endpoints: number of endpoints (if multitasking) or classes
        (if classification) to predict, defaults to 1

    :param str loss_fn: loss function to optimize, defaults to 'bce'

    :param dict loss_args: additional arguments for the loss function, defaults to {}

    :param str optimizer: optimizer to use while training, defaults to 'adamw'

    :param dict optimizer_args: additional arguments for the optimizer, defaults
        to {'lr': 1e-4, 'weight_decay': 1e-4}

    :param str scheduler: learning rate scheduler, defaults to 'warmup_linear_decay'

    :param dict scheduler_args: additional arguments for the scheduler, defaults
        to {'min_lr': 1e-5}. If 'total_steps' is not provided, it is
        auto-computed as num_epochs * ceil(len(train_data) / batch_size).

    :param int num_epochs: number of epochs to train for, defaults to 100

    :param int batch_size: batch size for training and prediction, defaults to 64

    :param bool stochastic_weight_averaging: whether to add SWA epochs after
        regular training, defaults to False

    :param bool early_stopping: whether to use early stopping, defaults to True

    :param int patience: how many epochs to wait before early stopping, defaults to 10

    :param int devices: number of devices for training, defaults to 1

    :param str accelerator: hardware accelerator ('cpu', 'gpu', 'tpu', or 'hpu'),
        defaults to 'gpu'

    :param int max_length: maximum SMILES length allowed, defaults to 200

    :param int num_augmentations: number of SMILES augmentations for training,
        defaults to 7

    :param list[str] | None feature_list: list of additional descriptor sets to
        compute, defaults to None

    :param dict label_encoder_params: parameters for the label encoder, defaults to {}

    :param str | list[str] | dict | None label_transform_map: label transform
        configuration, defaults to None

    :param bool augment_resonance: whether to augment with resonance structures,
        defaults to False

    :param int seed: random seed for reproducibility, defaults to 0
    """

    def __init__(
        self,
        enc_hidden_dim: int = 256,
        enc_kernel_dims: list[int] = [3, 5, 8, 11, 14],
        enc_num_heads: int = 8,
        enc_activation: str = "swish",
        enc_dropout: float = 0.2,
        pred_hidden_dims: list[int] = [256, 256],
        pred_task_head_dims: list[int] | None = None,
        pred_activation: str = "swish",
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
        patience: int = 10,
        devices: int = 1,
        accelerator: str = "gpu",
        max_length: int = 200,
        num_augmentations: int = 7,
        feature_list: list[str] | None = None,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        augment_resonance: bool = False,
        seed: int = 0,
    ):
        self._architecture = CNNModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(CNNClassifier, self).__init__(params)


@ScikitLearnModelRegistry.register()
class CNNRegressor(BaseScikitLearnCLM, ScikitLearnRegressorMixin):
    """1D convolutional neural network (CNN) for molecular property regression.

    Learns to embed chemical language (e.g. SMILES) via convolutional layers
    and predict properties in an end-to-end manner. Only compatible with
    regression datasets.
    Inherits from :class:`~matcha.sklearn.clm.base_sklearn_clm.BaseScikitLearnCLM`.

    References:

    - Boldini et al., *J. Cheminform.* (2024): https://arxiv.org/abs/2407.12152
    - Boldini et al., *Digital Discovery* (2023):
      https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g

    Example usage:

    .. code-block:: python

        model = CNNRegressor()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_hidden_dim: number of filters per convolutional layer,
        defaults to 256

    :param list[int] enc_kernel_dims: kernel sizes for each convolutional layer,
        defaults to [3, 5, 8, 11, 14]

    :param int enc_num_heads: number of attention heads for computing self-attention,
        defaults to 8

    :param str enc_activation: activation function to use throughout the encoder,
        defaults to 'swish'

    :param float enc_dropout: dropout rate across the encoder, defaults to 0.2

    :param list[int] pred_hidden_dims: shape of hidden MLP layers in the predictor,
        defaults to [256, 256]

    :param list[int] | None pred_task_head_dims: shape of the MLP layers
        dedicated to each task independently, defaults to None

    :param str pred_activation: activation function to use in the predictor, defaults
        to 'swish'

    :param float pred_dropout: dropout rate across the predictor, defaults to 0.2

    :param int num_endpoints: number of endpoints to predict (for multitasking),
        defaults to 1

    :param str loss_fn: loss function to optimize, defaults to 'mse'

    :param dict loss_args: additional arguments for the loss function, defaults to {}

    :param str optimizer: optimizer to use while training, defaults to 'adamw'

    :param dict optimizer_args: additional arguments for the optimizer, defaults
        to {'lr': 1e-4, 'weight_decay': 1e-4}

    :param str scheduler: learning rate scheduler, defaults to 'warmup_linear_decay'

    :param dict scheduler_args: additional arguments for the scheduler, defaults
        to {'min_lr': 1e-5}. If 'total_steps' is not provided, it is
        auto-computed as num_epochs * ceil(len(train_data) / batch_size).

    :param int num_epochs: number of epochs to train for, defaults to 100

    :param int batch_size: batch size for training and prediction, defaults to 64

    :param bool stochastic_weight_averaging: whether to add SWA epochs after
        regular training, defaults to False

    :param bool early_stopping: whether to use early stopping, defaults to True

    :param int patience: how many epochs to wait before early stopping, defaults to 10

    :param int devices: number of devices for training, defaults to 1

    :param str accelerator: hardware accelerator ('cpu', 'gpu', 'tpu', or 'hpu'),
        defaults to 'gpu'

    :param int max_length: maximum SMILES length allowed, defaults to 200

    :param int num_augmentations: number of SMILES augmentations for training,
        defaults to 7

    :param list[str] | None feature_list: list of additional descriptor sets to
        compute, defaults to None

    :param bool clip: whether to clip predictions to the training label range,
        defaults to True

    :param dict label_encoder_params: parameters for the label encoder, defaults to {}

    :param str | list[str] | dict | None label_transform_map: label transform
        configuration, defaults to None

    :param str scaler_type: type of feature scaler to use, defaults to 'standard'

    :param bool augment_resonance: whether to augment with resonance structures,
        defaults to False

    :param int seed: random seed for reproducibility, defaults to 0
    """

    def __init__(
        self,
        enc_hidden_dim: int = 256,
        enc_kernel_dims: list[int] = [3, 5, 8, 11, 14],
        enc_num_heads: int = 8,
        enc_activation: str = "swish",
        enc_dropout: float = 0.2,
        pred_hidden_dims: list[int] = [256, 256],
        pred_task_head_dims: list[int] | None = None,
        pred_activation: str = "swish",
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
        patience: int = 10,
        devices: int = 1,
        accelerator: str = "gpu",
        max_length: int = 200,
        num_augmentations: int = 7,
        feature_list: list[str] | None = None,
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        scaler_type: str = "standard",
        augment_resonance: bool = False,
        seed: int = 0,
    ):
        self._architecture = CNNModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(CNNRegressor, self).__init__(params)
