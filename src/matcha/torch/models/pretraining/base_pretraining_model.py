"""Base class for pretraining models."""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch
import lightning as L

from matcha.nn.losses import LossRegistry
from matcha.nn.optimizers import OptimizerRegistry
from matcha.nn.schedulers import SchedulerRegistry
from matcha.utils.registry import ClassRegistry
from matcha.utils import silence_nuisance_warnings


class BasePretrainingModel(L.LightningModule, ABC):
    """Base class for all pretraining models.

    This class provides common functionality for pretraining models,
    including loss function parsing, optimizer/scheduler configuration, and
    training/validation step logic.

    Subclasses must implement:
        - forward(): Returns predictions for the pretraining task
        - encode(): Returns learned representations
    """

    def __init__(self):
        """Initialize base pretraining model with null encoder and prediction head."""
        super().__init__()
        silence_nuisance_warnings()
        self.encoder = None
        self.prediction_head = None

    def _parse_train_config(self):
        """Parse training configuration from hyperparameters."""
        self.optimizer = OptimizerRegistry[self.hparams["optimizer"]](
            self.parameters(), **self.hparams["optimizer_args"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.scheduler = SchedulerRegistry[self.hparams["scheduler"]](
                self.optimizer, **self.hparams["scheduler_args"]
            )

        self.loss_fn = LossRegistry[self.hparams["loss_fn"]](
            **self.hparams.get("loss_args", {})
        )

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Forward pass for pretraining.

        :param batch: Input batch
        :return: Predictions for the pretraining task
        """
        pass

    @abstractmethod
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Extract learned representations.

        :param batch: Input batch
        :return: Learned representations
        """
        pass

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step logic.

        :param batch: Input batch
        :param batch_idx: Batch index
        :return: Loss value
        """
        y = batch["y"]
        y_pred = self.forward(batch)
        train_loss = self.loss_fn(y_pred, y)
        self.log("train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step logic.

        :param batch: Input batch
        :param batch_idx: Batch index
        :return: Loss value
        """
        y = batch["y"]
        y_pred = self.forward(batch)
        val_loss = self.loss_fn(y_pred, y)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return val_loss

    def predict_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Prediction step logic.

        :param batch: Input batch
        :return: Predictions
        """
        return self.forward(batch)

    def configure_optimizers(self):
        """Configure optimizers and schedulers for Lightning Trainer."""
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Encoder export for downstream finetuning
    # ------------------------------------------------------------------

    def export_encoder_checkpoint(self, path: str) -> None:
        """Save only the encoder weights for downstream finetuning.

        Writes ``encoder.ckpt`` containing the encoder's ``state_dict``
        into the given directory.  This file is consumed by
        :class:`PretrainedEncoderWrapper` when a finetuner loads a
        pretrained artifact with ``origin_type == "pretraining"``.

        :param str path: directory where ``encoder.ckpt`` will be written
        """
        if self.encoder is None:
            raise RuntimeError("Cannot export encoder — self.encoder is None.")
        os.makedirs(path, exist_ok=True)
        encoder_path = os.path.join(path, "encoder.ckpt")
        torch.save(self.encoder.state_dict(), encoder_path)


PretrainingModelRegistry = ClassRegistry[BasePretrainingModel]()
