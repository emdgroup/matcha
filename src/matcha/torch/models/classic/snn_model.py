"""Self-Normalizing Neural Network (SNN) classic model for tabular molecular features."""

from typing import Any
from matcha.torch.models.classic.base_classic_model import (
    BaseClassicModel,
    ClassicModelRegistry,
)
from matcha.torch.predictors.snn import SNN
from matcha.nn.deep_lasso import deep_lasso_regularizer
from matcha.nn.losses import MultiLoss
from matcha.utils.schemas import SNNInputModel

from lightning.pytorch.core.mixins import HyperparametersMixin


@ClassicModelRegistry.register()
class SNNModel(BaseClassicModel, HyperparametersMixin):
    """Self-Normalizing Neural Network (SNN) to predict molecular properties from molecular
    descriptors and fingerprints. SNNs use SELU activations and AlphaDropout to maintain
    self-normalizing properties, which can help with training deep networks without batch
    normalization.

    It inherits from :class:`BaseClassicModel` for common training and predicting routines
    (e.g. forward pass) and from :class:`lightning.pytorch.core.mixins` for saving its
    hyperparameters.

    References:
    - Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515

    Example usage:

    .. code-block:: python
        model = SNNModel()
        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param list[int] hidden_dims: shape of hidden SNN layers in the predictor,
        defaults to [512, 512, 128]

    :param float dropout: dropout rate across the predictor (uses AlphaDropout),
        defaults to 0.05

    :param int num_endpoints: number of endpoints (if multitasking) or classes,
        (if classification) to predict, defaults to 1

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

    :param int num_parallel: number of parallel heads in MultiLn layers, defaults to 8
    """

    def __init__(
        self,
        additional_mol_features_dim: int = 0,
        hidden_dims: list[int] = [512, 512, 128],
        num_parallel: int = 8,
        dropout: float = 0.05,
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
        self.params = SNNInputModel(**self.hparams)
        self.deep_lasso_weight = deep_lasso_weight
        self.predictor = SNN(
            self._get_predictor_input_dim(),
            hidden_dims,
            num_endpoints,
            dropout,
            num_parallel,
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

        if not isinstance(self.loss_fn, MultiLoss):
            train_loss = self.loss_fn(y_pred, y)
        else:
            # MultiLoss always returns (loss, per-task log); training-only per-task
            # logging is deliberately dropped here to keep the SNN training step
            # minimal. See issue #41.
            train_loss, _ = self.loss_fn(y_pred, y, self.global_step)

        if self.deep_lasso_weight > 0:
            reg = deep_lasso_regularizer(train_loss, mol_features)
            train_loss = (
                self.deep_lasso_weight * reg + (1 - self.deep_lasso_weight) * train_loss
            )

        self.log("train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True)

        return train_loss
