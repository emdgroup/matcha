"""Identity-only predictor for :class:`Finetuner` with ``keep_existing_predictor=False``.

When a :class:`Finetuner` is asked to discard the pretrained model's predictor,
every ``.predictor`` in the ``self.pretrain`` chain — including any nested
:class:`Finetuner` wrappers and the leaf classic/pretraining model — is
replaced with a :class:`PassthroughPredictor`. The result: ``self.pretrain.encode(batch)``
chains through with identity behavior at each level, and the new
``pred_hidden_dims`` MLP consumes the leaf encoder's output directly.

This module mirrors the role played by :class:`~pretrained_encoder_wrapper._StubPredictor`
for pure ``origin_type="pretraining"`` artifacts: both provide the minimal
attribute surface (:attr:`prediction_head`) that :class:`Finetuner` and
:class:`~matcha.torch.models.classic.base_classic_model.BaseClassicModel`
expect. :class:`PassthroughPredictor` additionally exposes ``.encode`` and
``.forward`` because :meth:`BaseClassicModel.encode` dispatches through
``predictor.encode`` — using :class:`torch.nn.Identity` would raise
``AttributeError``.
"""

from __future__ import annotations

import torch
from torch import nn


class PassthroughPredictor(nn.Module):
    """Identity predictor used when the pretrained predictor is discarded.

    The parameter is named ``mol_features`` because
    :meth:`~matcha.torch.models.classic.base_classic_model.BaseClassicModel._unpack_batch_and_call`
    uses signature-based dispatch and looks up ``batch["mol_features"]``.
    """

    def __init__(self):
        super().__init__()
        self.prediction_head = None

    def encode(self, mol_features: torch.Tensor) -> torch.Tensor:
        return mol_features

    def forward(self, mol_features: torch.Tensor) -> torch.Tensor:
        return mol_features
