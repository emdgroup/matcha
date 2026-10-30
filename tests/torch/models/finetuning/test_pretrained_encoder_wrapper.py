"""Tests for :class:`matcha.torch.models.finetuning.PretrainedEncoderWrapper`.

Regression coverage for the MLM path: the wrapper must return a flat
``[batch_size, hidden_dim]`` embedding when adapting an MLM encoder. An
earlier bug called ``encoder(token_ids)`` (which already CLS-pools for the
classic path) and then sliced ``[:, 0, :]`` on the pooled 2D tensor, raising
``IndexError: too many indices for tensor of dimension 2`` on the first
finetuning step. The fix routes through ``encoder.forward_tokens`` so the
wrapper is the sole owner of the CLS slice.
"""

from unittest.mock import MagicMock

import pytest
import torch

from matcha.nn.losses import MultiLoss
from matcha.torch.encoders.roformer import RoFormer
from matcha.torch.models.finetuning.finetuner import Finetuner
from matcha.torch.models.finetuning.pretrained_encoder_wrapper import (
    PretrainedEncoderWrapper,
)
from matcha.torch.models.mixin import ModelMixin


_VOCAB_SIZE = 32
_HIDDEN_DIM = 16
_EXPANSION_DIM = 32
_NUM_HEADS = 2
_NUM_LAYERS = 2


def _make_roformer() -> RoFormer:
    return RoFormer(
        num_characters=_VOCAB_SIZE,
        hidden_dim=_HIDDEN_DIM,
        expansion_dim=_EXPANSION_DIM,
        num_heads=_NUM_HEADS,
        num_layers=_NUM_LAYERS,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


def test_mlm_encode_returns_flat_embedding():
    """MLM-typed wrapper must CLS-pool to ``[batch_size, hidden_dim]``."""
    torch.manual_seed(0)
    encoder = _make_roformer()
    encoder.eval()
    wrapper = PretrainedEncoderWrapper(encoder, encoder_type="mlm")

    batch_size, seq_len = 3, 7
    tokens = torch.randint(low=1, high=_VOCAB_SIZE, size=(batch_size, seq_len))

    with torch.no_grad():
        embedding = wrapper.encode({"token_ids": tokens})

    assert embedding.shape == (batch_size, _HIDDEN_DIM)


def test_mlm_encode_matches_forward_tokens_cls_slice():
    """The wrapper's MLM output must equal ``forward_tokens(x)[:, 0, :]``.

    Structural check that the wrapper owns the CLS slice and does not
    double-pool by calling the encoder's classic ``forward``.
    """
    torch.manual_seed(0)
    encoder = _make_roformer()
    encoder.eval()
    wrapper = PretrainedEncoderWrapper(encoder, encoder_type="mlm")

    tokens = torch.randint(low=1, high=_VOCAB_SIZE, size=(2, 5))
    with torch.no_grad():
        expected = encoder.forward_tokens(tokens)[:, 0, :]
        actual = wrapper.encode({"token_ids": tokens})

    assert torch.allclose(actual, expected, atol=1e-6)


class TestFinetunerMultiLoss:
    """Regression for issue #48: ``Finetuner.training_step`` must consume
    ``MultiLoss``'s ``(total_loss, loss_log)`` tuple return without raising
    ``AttributeError: 'tuple' object has no attribute 'backward'``.

    Mirrors the pattern used in ``tests/nn/test_losses.py`` — a
    ``MagicMock(spec=...)`` with a bound method and a real ``MultiLoss``
    in train mode, avoiding any need for a pretrained artifact on disk.
    """

    @pytest.fixture()
    def loss_configs(self):
        return [
            {
                "loss_fn": "mse",
                "loss_args": {},
                "task_map": [0, 1],
                "init_w": 1.0,
                "final_w": 0.5,
                "T": 10,
                "warmup": 2,
            },
            {
                "loss_fn": "mae",
                "loss_args": {},
                "task_map": [2],
                "init_w": 0.5,
                "final_w": 1.0,
                "T": 10,
                "warmup": 0,
            },
        ]

    def _make_finetuner_mock(self, loss_configs, *, with_pretrain_optimizer: bool):
        model = MagicMock(spec=Finetuner)
        model.training_step = Finetuner.training_step.__get__(model)
        model.global_step = 0

        preds = torch.randn(8, 3, requires_grad=True)
        model.forward = MagicMock(return_value=preds)

        loss_fn = MultiLoss(loss_configs)
        loss_fn.train()
        model.loss_fn = loss_fn

        model.manual_backward = MagicMock()
        model.predictor_optimizer = MagicMock()
        model.predictor_scheduler = MagicMock()
        model.pretrain_optimizer = MagicMock() if with_pretrain_optimizer else None

        if with_pretrain_optimizer:
            # Regression for issue #52: `training_step` must route `.step()`
            # through Lightning-wrapped optimizers so `global_step` advances
            # and the `MultiLoss` weight curriculum interpolates.
            wrapped_predictor = MagicMock()
            wrapped_pretrain = MagicMock()

            def _advance_global_step(*_args, **_kwargs):
                model.global_step += 1

            wrapped_predictor.step = MagicMock(side_effect=_advance_global_step)
            wrapped_pretrain.step = MagicMock()
            model.optimizers = MagicMock(
                return_value=[wrapped_predictor, wrapped_pretrain]
            )
            model._wrapped_predictor = wrapped_predictor
            model._wrapped_pretrain = wrapped_pretrain

        logged: dict = {}

        def fake_log(name, value, **_kwargs):
            logged[name] = value.item() if isinstance(value, torch.Tensor) else value

        model.log = MagicMock(side_effect=fake_log)
        model._logged = logged
        return model

    def test_training_step_multiloss(self, loss_configs):
        """Full fine-tuning path: dual optimizers + manual_backward on the
        MultiLoss tuple must receive a scalar tensor, not the tuple itself.

        Also regression for issue #52: `.step()` must be routed through
        `self.optimizers()` (Lightning-wrapped optimizers) so
        `manual_optimization.optim_step_progress` advances and the
        `MultiLoss` weight curriculum interpolates off `init_w`.
        """
        model = self._make_finetuner_mock(loss_configs, with_pretrain_optimizer=True)
        batch = {"y": torch.randn(8, 3)}

        # First step — issue #48 regression checks (tuple unpacking).
        train_loss = model.training_step(batch, 0)

        assert isinstance(train_loss, torch.Tensor)
        assert train_loss.dim() == 0
        assert train_loss.requires_grad
        # Direct regression check: manual_backward saw a scalar Tensor, not a tuple.
        model.manual_backward.assert_called_once()
        (arg,) = model.manual_backward.call_args.args
        assert isinstance(arg, torch.Tensor)
        assert arg.dim() == 0
        # Logged keys: total + per-task loss + per-task weight.
        assert "train_loss" in model._logged
        assert any(
            k.startswith("train_task_") and k.endswith("_loss") for k in model._logged
        )
        weight_keys = [
            k
            for k in model._logged
            if k.startswith("train_task_") and k.endswith("_weight")
        ]
        assert weight_keys
        init_weights = {k: model._logged[k] for k in weight_keys}

        # Run enough further steps to pass warmup=2 and progress along T=10.
        for _ in range(9):
            model.training_step(batch, 0)

        # Issue #52 regression: `self.optimizers()` was consulted on every step
        # and wrapped `.step()` was invoked, while the raw torch refs were not.
        assert model.optimizers.call_count == 10
        assert model._wrapped_predictor.step.call_count == 10
        assert model._wrapped_pretrain.step.call_count == 10
        model.predictor_optimizer.step.assert_not_called()
        model.pretrain_optimizer.step.assert_not_called()
        # Simulated `global_step` counter incremented via the wrapped-step
        # side-effect, mirroring Lightning's `optim_step_progress` advance.
        assert model.global_step == 10
        # At least one task weight has moved off its `init_w` because the
        # `MultiLoss` schedule now receives a non-zero `T_current`.
        final_weights = {k: model._logged[k] for k in weight_keys}
        assert any(
            not torch.isclose(
                torch.tensor(final_weights[k]),
                torch.tensor(init_weights[k]),
            )
            for k in weight_keys
        )

    def test_training_step_multiloss_lora_path(self, loss_configs):
        """LoRA path (``pretrain_optimizer is None``): manual-optimization
        block is skipped, but MultiLoss logging and scalar return still work.
        """
        model = self._make_finetuner_mock(loss_configs, with_pretrain_optimizer=False)
        batch = {"y": torch.randn(8, 3)}

        train_loss = model.training_step(batch, 0)

        assert isinstance(train_loss, torch.Tensor)
        assert train_loss.dim() == 0
        assert train_loss.requires_grad
        model.manual_backward.assert_not_called()
        model.predictor_scheduler.step.assert_not_called()
        assert "train_loss" in model._logged
        assert any(
            k.startswith("train_task_") and k.endswith("_loss") for k in model._logged
        )

    def test_validation_step_multiloss(self, loss_configs):
        """``Finetuner`` inherits ``ModelMixin.validation_step``, which already
        handles ``MultiLoss``. Guards against a future override losing the
        tuple-unpacking branch.
        """
        model = MagicMock(spec=Finetuner)
        model.validation_step = ModelMixin.validation_step.__get__(model)
        model.global_step = 0
        model._max_task_tracking_n = 100
        model._label_names = []

        preds = torch.randn(8, 3, requires_grad=True)
        model.forward = MagicMock(return_value=preds)

        loss_fn = MultiLoss(loss_configs)
        loss_fn.train()
        model.loss_fn = loss_fn

        logged: dict = {}

        def fake_log(name, value, **_kwargs):
            logged[name] = value.item() if isinstance(value, torch.Tensor) else value

        model.log = MagicMock(side_effect=fake_log)

        # Regression targets are floats so the per-task metrics branch is exercised.
        batch = {"y": torch.randn(8, 3)}
        result = model.validation_step(batch, 0)

        assert "val_loss" in result
        assert isinstance(result["val_loss"], torch.Tensor)
        assert result["val_loss"].dim() == 0
        assert "val_loss" in logged
        assert any(k.startswith("val_task_") and k.endswith("_loss") for k in logged)
