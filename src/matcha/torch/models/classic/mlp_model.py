"""Multi-Layer Perceptron (MLP) classic model for tabular molecular features."""

from typing import Any
from matcha.torch.models.classic.base_classic_model import (
    BaseClassicModel,
    ClassicModelRegistry,
)
from matcha.torch.predictors.mlp import MLP
from matcha.nn.deep_lasso import deep_lasso_regularizer
from matcha.nn.losses import MultiLoss
from matcha.utils.schemas import MLPInputModel

from lightning.pytorch.core.mixins import HyperparametersMixin


@ClassicModelRegistry.register()
class MLPModel(BaseClassicModel, HyperparametersMixin):
    """Multi Layer Perceptron (MLP) to predict molecular properties from molecular
    descriptors and fingerprints. Additionally, this class includes deep lasso regularization
    and custom linear layer stacks to further improve performance.
    It inherits from :class:`BaseClassicModel`
    for common training and predicting routines (e.g. forward pass) and from
    :class:`lightning.pytorch.core.mixins` for saving its hyperparameters.
    References:
    - https://arxiv.org/pdf/2311.05877
    - https://docs.fast.ai/layers.html#linbndrop

    Example usage:

    .. code-block:: python
        model = MLPModel()
        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param list[int] hidden_dims: shape of hidden MLP layers in the predictor,
        defaults to [512, 512, 128]

    :param str activation: activation function to use across the network, defaults
        to 'swish'

    :param float dropout: dropout rate across the encoder and predictor, defaults to
        0.2

    :param int num_endpoints: number of endpoints (if multitasking) or classes,
        (if classification) to predict, defaults to 1

    :param bool batchnorm: whether to use batchnorm across all layers, defaults
        to True

    :param float deep_lasso_weight: weight for the deep lasso regularization, defaults
        to 0.1

    :param str loss_fn: loss function to optimize, defaults to 'mse'

    :param dict loss_args: additional arguments for the loss function (e.g. for
        focal loss), defaults to {}

    :param str optimizer: optimizer to use while training, defaults to 'adam'

    :param dict optimizer_args: additional arguments for the optimizer, defaults
        to {'lr': 1e-3}

    :param str scheduler: scheduler to change the learning rate during training,
        defaults to 'cosine_annealing'

    :param dict scheduler_args: additional arguments for the scheduler, defaults
        to {'min_lr': 1e-6, 'total_steps': 50}
    """

    def __init__(
        self,
        additional_mol_features_dim: int = 0,
        hidden_dims: list[int] = [512, 512, 128],
        task_head_dims: list[int] | None = None,
        dropout: float = 0.2,
        activation: str = "swish",
        num_endpoints: int = 1,
        deep_lasso_weight: float = 0.1,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 1e-3},
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {"min_lr": 1e-6, "total_steps": 50},
    ):
        super().__init__(additional_mol_features_dim)
        self.save_hyperparameters()
        self.params = MLPInputModel(**self.hparams)
        self.deep_lasso_weight = deep_lasso_weight
        self.predictor = MLP(
            self._get_predictor_input_dim(),
            hidden_dims,
            task_head_dims,
            num_endpoints,
            dropout,
            activation,
            "batch",
        )
        self.encoder = None
        self._parse_train_config()

    def training_step(self, batch: dict[str, Any], batch_idx):
        """Training step with deep lasso regularisation.

        :param dict[str, Any] batch: batch containing 'mol_features' and 'y'
        :param int batch_idx: index of the current batch
        :returns: training loss
        :rtype: torch.Tensor
        """
        mol_features, y = batch["mol_features"], batch["y"]
        mol_features.requires_grad_()
        y_pred = self.forward(batch)
        train_loss = self.loss_fn(y_pred, y)
        if self.deep_lasso_weight > 0:
            reg = deep_lasso_regularizer(train_loss, mol_features)
            train_loss = (
                self.deep_lasso_weight * reg + (1 - self.deep_lasso_weight) * train_loss
            )

        if not isinstance(self.loss_fn, MultiLoss):
            train_loss = self.loss_fn(y_pred, y)
            self.log(
                "train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True
            )
        else:
            train_loss, loss_log = self.loss_fn(y_pred, y, self.global_step)
            self.log(
                "train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True
            )
            for name, log in loss_log.items():
                self.log(
                    f"train_{name}_loss",
                    log["loss"],
                    prog_bar=True,
                    on_step=True,
                    sync_dist=True,
                )
                self.log(
                    f"train_{name}_weight",
                    log["weight"],
                    prog_bar=True,
                    on_step=True,
                    sync_dist=True,
                )

        return train_loss
