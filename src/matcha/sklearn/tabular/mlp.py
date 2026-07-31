"""Sklearn-compatible MLP wrappers for molecular property prediction from tabular descriptors."""

from matcha.sklearn.tabular.base_sklearn_tabular import BaseScikitLearnTabular
from matcha.torch.models.classic import MLPModel
from matcha.sklearn.base_sklearn_model import (
    ScikitLearnModelRegistry,
    ScikitLearnClassifierMixin,
    ScikitLearnRegressorMixin,
)


@ScikitLearnModelRegistry.register()
class MLPClassifier(BaseScikitLearnTabular, ScikitLearnClassifierMixin):
    """Multi-Layer Perceptron (MLP) for molecular property classification from tabular features.

    Predicts molecular properties from molecular descriptors and fingerprints.
    Includes deep lasso regularization and LinBnDrop linear layer stacks.
    Only compatible with classification datasets.
    Inherits from :class:`~matcha.sklearn.tabular.base_sklearn_tabular.BaseScikitLearnTabular`.

    References:

    - Boldini et al., *J. Cheminform.* (2024): https://arxiv.org/abs/2311.05877
    - fast.ai LinBnDrop: https://docs.fast.ai/layers.html#linbndrop

    Example usage:

    .. code-block:: python

        model = MLPClassifier()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param list[int] hidden_dims: shape of hidden MLP layers, defaults to [512, 256]

    :param list[int] | None task_head_dims: shape of per-task MLP head layers,
        defaults to None

    :param str activation: activation function, defaults to 'mish'

    :param float dropout: dropout rate across the network, defaults to 0.2

    :param int num_endpoints: number of endpoints (if multitasking) or classes
        (if classification) to predict, defaults to 1

    :param float deep_lasso_weight: weight for deep lasso regularization,
        defaults to 0.1

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

    :param int patience: how many epochs to wait before early stopping, defaults to 20

    :param int devices: number of devices for training, defaults to 1

    :param str accelerator: hardware accelerator ('cpu', 'gpu', 'tpu', or 'hpu'),
        defaults to 'gpu'

    :param list[str] feature_list: descriptor/fingerprint sets to use as input,
        defaults to ['ECFP_count', 'rdkit_all_descriptors']

    :param dict label_encoder_params: parameters for the label encoder, defaults to {}

    :param str | list[str] | dict | None label_transform_map: label transform
        configuration, defaults to None

    :param bool augment_resonance: whether to augment with resonance structures,
        defaults to False

    :param int seed: random seed for reproducibility, defaults to 0
    """

    def __init__(
        self,
        hidden_dims: list[int] = [512, 256],
        task_head_dims: list[int] | None = None,
        activation: str = "mish",
        dropout: float = 0.2,
        num_endpoints: int = 1,
        deep_lasso_weight: float = 0.1,
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
        feature_list: list[str] = ["ECFP_count", "rdkit_all_descriptors"],
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        augment_resonance: bool = False,
        seed: int = 0,
    ):
        self._architecture = MLPModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(MLPClassifier, self).__init__(params)


@ScikitLearnModelRegistry.register()
class MLPRegressor(BaseScikitLearnTabular, ScikitLearnRegressorMixin):
    """Multi-Layer Perceptron (MLP) for molecular property regression from tabular features.

    Predicts molecular properties from molecular descriptors and fingerprints.
    Includes deep lasso regularization and LinBnDrop linear layer stacks.
    Only compatible with regression datasets.
    Inherits from :class:`~matcha.sklearn.tabular.base_sklearn_tabular.BaseScikitLearnTabular`.

    References:

    - Boldini et al., *J. Cheminform.* (2024): https://arxiv.org/abs/2311.05877
    - fast.ai LinBnDrop: https://docs.fast.ai/layers.html#linbndrop

    Example usage:

    .. code-block:: python

        model = MLPRegressor()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param list[int] hidden_dims: shape of hidden MLP layers, defaults to [512, 256]

    :param list[int] | None task_head_dims: shape of per-task MLP head layers,
        defaults to None

    :param str activation: activation function, defaults to 'mish'

    :param float dropout: dropout rate across the network, defaults to 0.2

    :param int num_endpoints: number of endpoints to predict (for multitasking),
        defaults to 1

    :param float deep_lasso_weight: weight for deep lasso regularization,
        defaults to 0.1

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

    :param int patience: how many epochs to wait before early stopping, defaults to 20

    :param int devices: number of devices for training, defaults to 1

    :param str accelerator: hardware accelerator ('cpu', 'gpu', 'tpu', or 'hpu'),
        defaults to 'gpu'

    :param list[str] feature_list: descriptor/fingerprint sets to use as input,
        defaults to ['ECFP_count', 'rdkit_all_descriptors']

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
        hidden_dims: list[int] = [512, 256],
        task_head_dims: list[int] | None = None,
        activation: str = "mish",
        dropout: float = 0.2,
        num_endpoints: int = 1,
        deep_lasso_weight: float = 0.1,
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
        feature_list: list[str] = ["ECFP_count", "rdkit_all_descriptors"],
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        scaler_type: str = "standard",
        augment_resonance: bool = False,
        seed: int = 0,
    ):
        self._architecture = MLPModel
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super(MLPRegressor, self).__init__(params)
