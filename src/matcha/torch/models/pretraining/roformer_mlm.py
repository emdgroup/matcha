"""RoFormer variant for Masked Language Modeling (MLM)."""

from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.torch.encoders.roformer import RoFormer
from matcha.torch.models.pretraining.base_pretraining_model import (
    BasePretrainingModel,
    PretrainingModelRegistry,
)
from matcha.nn.layers import LnBnDr


@PretrainingModelRegistry.register()
class RoFormerMLM(BasePretrainingModel, HyperparametersMixin):
    """RoFormer model variant for Masked Language Modeling (MLM).

    This model returns per-token predictions for use in MLM self-supervised
    learning setups. Instead of aggregating token representations to a single
    vector, it computes predictions for each token position.

    The model can be used to predict masked tokens during pretraining, and the
    learned encoder can be transferred to downstream tasks.

    Example usage:

    .. code-block:: python

        model = RoFormerMLM(enc_num_characters=100)
        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param int enc_num_characters: Vocabulary size
    :param int enc_hidden_dim: Token embedding dimensionality, defaults to 256
    :param int enc_expansion_dim: Dimensionality inside transformer blocks, defaults to 1024
    :param int enc_num_heads: Number of attention heads, defaults to 4
    :param int enc_num_layers: Number of transformer blocks, defaults to 2
    :param float enc_attention_dropout: Dropout for attention, defaults to 0.1
    :param float enc_hidden_dropout: Dropout in dense layers, defaults to 0.1
    :param list[int] pred_hidden_dims: Hidden dimensions for prediction head, defaults to [512]
    :param str pred_activation: Activation function, defaults to 'gelu'
    :param float pred_dropout: Dropout for prediction head, defaults to 0.1
    :param str loss_fn: Loss function to use, defaults to 'cross_entropy'
    :param dict loss_args: Additional arguments for loss function
    :param str optimizer: Optimizer to use, defaults to 'adamw'
    :param dict optimizer_args: Additional arguments for optimizer
    :param str scheduler: Learning rate scheduler, defaults to 'cosine_annealing'
    :param dict scheduler_args: Additional arguments for scheduler
    """

    def __init__(
        self,
        enc_num_characters: int,
        enc_hidden_dim: int = 256,
        enc_expansion_dim: int = 1024,
        enc_num_heads: int = 4,
        enc_num_layers: int = 2,
        enc_attention_dropout: float = 0.1,
        enc_hidden_dropout: float = 0.1,
        pred_hidden_dims: list[int] = [512],
        pred_activation: str = "gelu",
        pred_dropout: float = 0.1,
        loss_fn: str = "cross_entropy",
        loss_args: dict = {},
        optimizer: str = "adamw",
        optimizer_args: dict = {"lr": 1e-4},
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {"min_lr": 1e-6, "total_steps": 50},
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build encoder using the canonical RoFormer; pretraining consumes
        # per-token embeddings via ``encoder.forward_tokens``.
        self.encoder = RoFormer(
            num_characters=enc_num_characters,
            hidden_dim=enc_hidden_dim,
            expansion_dim=enc_expansion_dim,
            num_heads=enc_num_heads,
            num_layers=enc_num_layers,
            attention_dropout=enc_attention_dropout,
            hidden_dropout=enc_hidden_dropout,
        )

        # Build prediction head for MLM
        # Maps from hidden_dim -> vocab_size for each token position
        layers = []
        dim_list = [enc_hidden_dim] + pred_hidden_dims
        for i in range(len(dim_list) - 1):
            layers.append(
                LnBnDr(
                    dim_list[i], dim_list[i + 1], pred_dropout, pred_activation, "layer"
                )
            )
        layers.append(nn.Linear(dim_list[-1], enc_num_characters))
        self.prediction_head = nn.Sequential(*layers)

        self._parse_train_config()

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Forward pass for MLM prediction.

        :param batch: Input batch containing 'token_ids' key
        :return: Logits for each token position [batch_size, seq_length, vocab_size]
        """
        token_ids = batch["token_ids"]
        # Get per-token embeddings: [batch_size, seq_length, hidden_dim]
        token_embeddings = self.encoder.forward_tokens(token_ids)
        # Predict vocabulary distribution for each token: [batch_size, seq_length, vocab_size]
        logits = self.prediction_head(token_embeddings)
        return logits

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Extract per-token representations.

        :param batch: Input batch containing 'token_ids' key
        :return: Per-token embeddings [batch_size, seq_length, hidden_dim]
        """
        token_ids = batch["token_ids"]
        return self.encoder.forward_tokens(token_ids)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for MLM.

        Expects batch to contain:
            - 'token_ids': Input tokens with some positions masked
            - 'y': Original token IDs (targets)
            - 'mask': Boolean mask indicating which positions to predict

        :param batch: Input batch
        :param batch_idx: Batch index
        :return: Loss value
        """
        y = batch["y"]  # [batch_size, seq_length]
        y_pred = self.forward(batch)  # [batch_size, seq_length, vocab_size]

        # Reshape for cross-entropy: [batch_size * seq_length, vocab_size] and [batch_size * seq_length]
        batch_size, seq_length, vocab_size = y_pred.shape
        y_pred_flat = y_pred.view(-1, vocab_size)
        y_flat = y.view(-1)

        # Apply mask if present to compute loss only on masked positions
        if "mask" in batch:
            mask = batch["mask"].view(-1)
            y_pred_flat = y_pred_flat[mask]
            y_flat = y_flat[mask]

        train_loss = self.loss_fn(y_pred_flat, y_flat)
        self.log("train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step for MLM.

        :param batch: Input batch
        :param batch_idx: Batch index
        :return: Loss value
        """
        y = batch["y"]
        y_pred = self.forward(batch)

        batch_size, seq_length, vocab_size = y_pred.shape
        y_pred_flat = y_pred.view(-1, vocab_size)
        y_flat = y.view(-1)

        if "mask" in batch:
            mask = batch["mask"].view(-1)
            y_pred_flat = y_pred_flat[mask]
            y_flat = y_flat[mask]

        val_loss = self.loss_fn(y_pred_flat, y_flat)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return val_loss
