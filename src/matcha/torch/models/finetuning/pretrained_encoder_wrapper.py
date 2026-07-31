"""Adapter that gives a pretraining encoder the classic-model interface.

The :class:`Finetuner` expects its ``self.pretrain`` object to expose:

* ``encoder``                     — the encoder module
* ``predictor.prediction_head``   — (set to ``None`` during init)
* ``encode(batch) -> Tensor``     — returns a flat ``[B, D]`` embedding
* ``latent_dim``                  — output dimensionality of the encoder
* ``_label_names``                — label name list (empty for pretrained)

Pretraining models (:class:`BaseGraphPretrainingModel`, :class:`RoFormerMLM`)
do not naturally satisfy this contract.  This thin wrapper bridges the gap
so that the ``Finetuner`` code does not need to know whether the upstream
model was trained via classic or pretraining.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _StubPredictor(nn.Module):
    """Minimal stand-in for the classic model's predictor attribute."""

    def __init__(self):
        super().__init__()
        self.prediction_head = None


class PretrainedEncoderWrapper(nn.Module):
    """Wraps a pretraining encoder to present the classic-model interface.

    :param nn.Module encoder: the encoder from a pretraining model
    :param str encoder_type: one of ``"graph"`` or ``"mlm"``
    """

    # Recognised encoder families
    _GRAPH_TYPE = "graph"
    _MLM_TYPE = "mlm"

    def __init__(self, encoder: nn.Module, encoder_type: str):
        super().__init__()
        if encoder_type not in (self._GRAPH_TYPE, self._MLM_TYPE):
            raise ValueError(
                f"encoder_type must be '{self._GRAPH_TYPE}' or "
                f"'{self._MLM_TYPE}', got '{encoder_type}'"
            )
        self.encoder = encoder
        self._encoder_type = encoder_type
        self.predictor = _StubPredictor()
        self._label_names: list[str] = []

    # ------------------------------------------------------------------
    # Interface expected by Finetuner
    # ------------------------------------------------------------------

    @property
    def latent_dim(self) -> int:
        return self.encoder.fp_dim

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return a flat ``[B, D]`` embedding, regardless of encoder family.

        * **Graph encoders** already aggregate via readout, returning
          ``[B, D]`` from their ``forward(graph)`` call.
        * **MLM encoders** return per-token embeddings ``[B, S, D]``;
          we CLS-pool (take index 0) to get ``[B, D]``.
        """
        if self._encoder_type == self._GRAPH_TYPE:
            graph = batch["graph"]
            return self.encoder(graph)

        # MLM path
        token_ids = batch["token_ids"]
        embeddings = self.encoder(token_ids)  # [B, S, D]
        return embeddings[:, 0, :]  # CLS pooling
