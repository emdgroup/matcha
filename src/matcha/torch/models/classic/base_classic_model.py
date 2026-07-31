"""Base class for classic (encoder + predictor) model architectures."""

import inspect
from typing import Any, Callable
import torch
from torch.utils.data import DataLoader
from matcha.nn.losses import MultiLoss
from matcha.nn.optimizers import OptimizerRegistry
from matcha.utils.registry import ClassRegistry
from matcha.torch.models.mixin import ModelMixin
from abc import ABC
from matcha.torch.predictors.mlp import MLP
from matcha.utils import silence_nuisance_warnings


class BaseClassicModel(ModelMixin, ABC):
    """Base class for all classic models. It is not meant to be instantiated directly,
    but rather to be used as a parent class for each classic model. It should encompass
    all basic routines needed for training and inference, therefore child classes only
    need to define what are the encoder and predictor attributes.
    """

    def __init__(self, additional_mol_features_dim: int = 0):
        """Initialise the base classic model.

        :param int additional_mol_features_dim: dimensionality of any extra
            molecular features concatenated to the encoder output before
            prediction, defaults to 0
        """
        super().__init__()
        silence_nuisance_warnings()
        self._mc_dropout_flag = False
        self._max_task_tracking_n = 100
        self._additional_mol_features_dim = additional_mol_features_dim
        self.encoder = None
        self.predictor = None
        self._label_names = []

    @property
    def additional_mol_features_dim(self) -> int:
        """Dimensionality of extra molecular features appended to the encoder output."""
        return self._additional_mol_features_dim

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the learned representation, i.e. the output of the
        hidden layers right before the prediction head."""
        return self.predictor.latent_dim

    def _get_predictor_input_dim(self) -> int:
        """Compute the input dimensionality of the predictor head.

        :returns: sum of the encoder fingerprint dimension and any additional
            molecular features dimension
        :rtype: int
        """
        predictor_input_dim = self.additional_mol_features_dim
        if self.encoder is not None:
            predictor_input_dim += self.encoder.fp_dim
        return predictor_input_dim

    def _parse_train_config(self):
        """Utility function to parse the string/dict inputs related to losses, optimizers and
        schedulers into proper torch objects.

        Will read the required arguments from self.hparams, as created by the
        HyperparametersMixin of the child class.

        If ``total_steps`` is required by the scheduler but not provided in
        ``scheduler_args``, a placeholder value of 1 is used. The real value
        is injected later by ``TrainingManager._maybe_inject_total_steps``
        once the training dataset size is known.
        """

        self.optimizer = OptimizerRegistry[self.hparams["optimizer"]](
            self.parameters(), **self.hparams["optimizer_args"]
        )

        self.scheduler = self._make_scheduler(
            self.hparams["scheduler"], self.optimizer, self.hparams["scheduler_args"]
        )

        self._parse_loss_fn(
            self.hparams["loss_fn"],
            self.hparams["loss_args"],
            self.hparams["num_endpoints"],
        )

        self._init_metric_containers()

    def _parse_predictor(self):
        """Utility function to parse the arguments related to the predictor.

        Will read the required arguments from self.hparams, as created by the
        HyperparametersMixin of the child class
        """
        self.predictor = MLP(
            input_dim=self._get_predictor_input_dim(),
            hidden_dims=self.hparams["pred_hidden_dims"],
            task_head_dims=self.hparams["pred_task_head_dims"],
            num_endpoints=self.hparams["num_endpoints"],
            dropout=self.hparams["pred_dropout"],
            activation=self.hparams["pred_activation"],
            norm="batch",
        )

    def _unpack_batch_and_call(
        self, batch: dict[str, Any], function: Callable[..., torch.Tensor]
    ) -> torch.Tensor:
        """Utility function to forward a batch through a given module.

        :param dict[str, Any] batch: batch of inputs to process

        :param Callable[..., torch.Tensor] function: function to call on the batch

        :return torch.Tensor: processed batch
        """
        forward_args = {}
        # Get the signature of the forward method
        sig = inspect.signature(function)

        # Iterate through the parameters
        for param in sig.parameters.values():
            if param.default == inspect.Parameter.empty:
                if param.name in batch:
                    forward_args[param.name] = batch[param.name]
                else:
                    raise ValueError(f"Missing required argument: {param.name}")
            else:
                if param.name in batch:
                    forward_args[param.name] = batch[param.name]
                else:
                    forward_args[param.name] = param.default

        return function(**forward_args)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Simple forward function for the training and prediction step. Depending
        on the architecture, the input can be in different formats.

        :param dict[str, Any] batch: batch of inputs to process

        :return torch.Tensor: batched predictions
        """

        # Collect the features from the encoder and the input mol features
        if self.encoder is not None:
            mol_features = self._unpack_batch_and_call(batch, self.encoder.forward)
            if "mol_features" in batch:
                mol_features = torch.cat([mol_features, batch["mol_features"]], dim=1)

            batch["mol_features"] = mol_features

        return self._unpack_batch_and_call(batch, self.predictor.forward)

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Simple encoding function for getting the latent space of a batch. Depending
        on the architecture, the input can be in different formats.

        :param dict[str, Any] batch: batch of inputs to process

        :return torch.Tensor: latent space
        """
        # Collect the features from the encoder and the input mol features
        if self.encoder is not None:
            mol_features = self._unpack_batch_and_call(batch, self.encoder.forward)
            if "mol_features" in batch:
                mol_features = torch.cat([mol_features, batch["mol_features"]], dim=1)

            batch["mol_features"] = mol_features
        return self._unpack_batch_and_call(batch, self.predictor.encode)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step logic for classic models.

        :param dict[str, Any] batch: batch of inputs to process

        :param int batch_idx: leftover from lightning tutorial which I am too scared
            to remove (TODO)

        :return torch.Tensor: batch loss
        """
        y = batch["y"]
        y_pred = self.forward(batch)
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

    def validation_step(self, batch, batch_idx) -> dict:
        super().validation_step(batch, batch_idx)

    def compute_learned_embedding(self, x: DataLoader) -> torch.Tensor:
        """Extracts learned embeddings for a given batch from a dataloader.

        :param object x: dataloader object containing the batched dataset to compute
            embeddings of

        :return torch.Tensor: learned embeddings for the dataset
        """
        output = []
        for batch in x:
            mol_features = self.encode(batch)
            output.append(mol_features)
        return output

    def predict_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Prediction step logic for classic models.

        :param dict[str, Any] batch: batch of inputs to process

        :return torch.Tensor: predictions for the batch
        """
        # decide whether to keep dropout on or off depending on flag
        if self.mc_dropout_flag:
            for module in self.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()
        else:
            for module in self.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
        return self.forward(batch)

    def configure_optimizers(self):
        """Configuration utility function to comply with the Lightning Trainer
        API
        """
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


ClassicModelRegistry = ClassRegistry[BaseClassicModel]()
