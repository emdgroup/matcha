"""Tests for multitask loss composition and GradNorm."""

from unittest.mock import MagicMock

import pytest
import torch

from matcha.nn.losses import GradNormLoss, MultitaskLoss, MultiLoss
from matcha.torch.models.classic.base_classic_model import BaseClassicModel
from matcha.torch.models.classic.mlp_model import MLPModel


# ===================================================================
# MultitaskLoss
# ===================================================================


class TestMultitaskLoss:
    def test_output_scalar(self, multitask_targets):
        loss_fn = MultitaskLoss(loss_fn="mse")
        preds = torch.randn_like(multitask_targets)
        loss = loss_fn(preds, multitask_targets.clone())
        assert loss.dim() == 0

    def test_handles_all_nan_column(self):
        """If an entire column is NaN, the loss should still be finite."""
        targets = torch.full((8, 3), float("nan"))
        targets[:, 0] = torch.randn(8)
        preds = torch.randn(8, 3)
        loss_fn = MultitaskLoss(loss_fn="mse")
        loss = loss_fn(preds, targets)
        assert torch.isfinite(loss)

    def test_grad_flows(self, multitask_targets):
        loss_fn = MultitaskLoss(loss_fn="mse")
        preds = torch.randn_like(multitask_targets, requires_grad=True)
        loss = loss_fn(preds, multitask_targets.clone())
        loss.backward()
        assert preds.grad is not None

    def test_per_task_losses_attribute_exists(self, multitask_targets):
        """After forward(), _per_task_losses should be set."""
        loss_fn = MultitaskLoss(loss_fn="mse")
        preds = torch.randn_like(multitask_targets)
        loss_fn(preds, multitask_targets.clone())
        assert hasattr(loss_fn, "_per_task_losses")

    def test_per_task_losses_shape(self, multitask_targets):
        """_per_task_losses shape should match [num_tasks]."""
        loss_fn = MultitaskLoss(loss_fn="mse")
        preds = torch.randn_like(multitask_targets)
        loss_fn(preds, multitask_targets.clone())
        assert loss_fn._per_task_losses.shape == (multitask_targets.shape[1],)

    def test_per_task_losses_detached(self, multitask_targets):
        """_per_task_losses should not require grad."""
        loss_fn = MultitaskLoss(loss_fn="mse")
        preds = torch.randn_like(multitask_targets, requires_grad=True)
        loss_fn(preds, multitask_targets.clone())
        assert not loss_fn._per_task_losses.requires_grad


# ===================================================================
# MultiLoss
# ===================================================================


class TestMultiLoss:
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

    def test_output_tuple_training(self, loss_configs):
        loss_fn = MultiLoss(loss_configs)
        loss_fn.train()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        result = loss_fn(preds, targets, T_current=0)
        # Training and eval share the always-tuple contract (see issue #41).
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, log = result
        assert loss.dim() == 0
        assert isinstance(log, dict)

    def test_output_tuple_eval(self, loss_configs):
        loss_fn = MultiLoss(loss_configs)
        loss_fn.eval()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        result = loss_fn(preds, targets, T_current=0)
        # In eval mode, returns (loss, log_dict)
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, log = result
        assert loss.dim() == 0
        assert isinstance(log, dict)

    def test_weight_during_warmup(self, loss_configs):
        loss_fn = MultiLoss(loss_configs)
        # T_current=0 is within warmup of first config (warmup=2)
        weight = loss_fn._calculate_weight(
            init_w=1.0, final_w=0.5, T=10, warmup=2, T_current=1
        )
        assert weight == 1.0  # Should be init_w during warmup

    def test_weight_after_warmup(self, loss_configs):
        loss_fn = MultiLoss(loss_configs)
        weight = loss_fn._calculate_weight(
            init_w=1.0, final_w=0.5, T=10, warmup=2, T_current=7
        )
        # Linear interpolation: progress=(7-2)/10=0.5, weight=1.0+0.5*(0.5-1.0)=0.75
        assert abs(weight - 0.75) < 1e-6

    def test_weight_after_completion(self, loss_configs):
        loss_fn = MultiLoss(loss_configs)
        weight = loss_fn._calculate_weight(
            init_w=1.0, final_w=0.5, T=10, warmup=2, T_current=20
        )
        assert weight == 0.5  # Should be final_w

    def test_handles_nan_targets(self, loss_configs):
        loss_fn = MultiLoss(loss_configs)
        loss_fn.train()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        targets[0, 0] = float("nan")
        targets[3, 2] = float("nan")
        loss, _ = loss_fn(preds, targets, T_current=0)
        assert torch.isfinite(loss)

    def _make_training_step_mock(self, spec_cls, loss_configs):
        """Build a MagicMock that runs the real training_step of ``spec_cls``.

        Follows the ``MagicMock(spec=...) + bound method`` pattern used in
        ``tests/pretraining/test_base_graph_pretraining.py``. Assigns a real
        MultiLoss in train mode and records ``self.log`` calls.
        """
        model = MagicMock(spec=spec_cls)
        model.training_step = spec_cls.training_step.__get__(model)
        model.global_step = 0
        model.deep_lasso_weight = 0.0

        preds = torch.randn(8, 3, requires_grad=True)
        model.forward = MagicMock(return_value=preds)

        loss_fn = MultiLoss(loss_configs)
        loss_fn.train()
        model.loss_fn = loss_fn

        logged: dict = {}

        def fake_log(name, value, **_kwargs):
            logged[name] = value.item() if isinstance(value, torch.Tensor) else value

        model.log = MagicMock(side_effect=fake_log)
        model._logged = logged
        return model

    def test_training_step_integration_base_classic_model(self, loss_configs):
        """Regression for issue #41: BaseClassicModel.training_step must
        consume MultiLoss's tuple return without ``iteration over a 0-d tensor``.
        """
        model = self._make_training_step_mock(BaseClassicModel, loss_configs)
        batch = {"y": torch.randn(8, 3)}

        train_loss = model.training_step(batch, 0)

        assert isinstance(train_loss, torch.Tensor)
        assert train_loss.dim() == 0
        assert train_loss.requires_grad
        assert "train_loss" in model._logged
        # At least one per-task loss and weight were logged.
        assert any(
            k.startswith("train_task_") and k.endswith("_loss") for k in model._logged
        )
        assert any(
            k.startswith("train_task_") and k.endswith("_weight") for k in model._logged
        )

    def test_training_step_integration_mlp_model(self, loss_configs):
        """Regression for issue #41: MLPModel.training_step must consume
        MultiLoss's tuple return without ``iteration over a 0-d tensor``.
        """
        model = self._make_training_step_mock(MLPModel, loss_configs)
        batch = {
            "mol_features": torch.randn(8, 4),
            "y": torch.randn(8, 3),
        }

        train_loss = model.training_step(batch, 0)

        assert isinstance(train_loss, torch.Tensor)
        assert train_loss.dim() == 0
        assert train_loss.requires_grad
        assert "train_loss" in model._logged
        assert any(
            k.startswith("train_task_") and k.endswith("_loss") for k in model._logged
        )
        assert any(
            k.startswith("train_task_") and k.endswith("_weight") for k in model._logged
        )


# ===================================================================
# GradNormLoss
# ===================================================================


class TestGradNormLoss:
    def test_output_scalar(self):
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0

    def test_initial_weights_are_ones(self):
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=4)
        assert torch.allclose(loss_fn.weights, torch.ones(4))

    def test_initial_losses_none(self):
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        assert loss_fn.initial_losses is None

    def test_initial_losses_set_after_forward(self):
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        _ = loss_fn(preds, targets)
        assert loss_fn.initial_losses is not None

    def test_reset_initial_losses(self):
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        _ = loss_fn(preds, targets)
        loss_fn.reset_initial_losses()
        assert loss_fn.initial_losses is None

    def test_handles_nan_targets(self):
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        targets[0, 1] = float("nan")
        loss = loss_fn(preds, targets)
        assert torch.isfinite(loss)

    def test_training_with_shared_layer(self):
        """In training mode with a shared layer, GradNorm should update weights."""
        shared_layer = torch.nn.Linear(16, 3)
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        loss_fn.train()

        x = torch.randn(8, 16, requires_grad=True)
        preds = shared_layer(x)
        targets = torch.randn(8, 3)

        initial_weights = loss_fn.weights.clone()
        loss = loss_fn(preds, targets, shared_layer=shared_layer)
        loss.backward()
        # Weights should have been updated
        assert not torch.allclose(loss_fn.weights, initial_weights)

    def test_eval_no_weight_update(self):
        """In eval mode, weights should not change."""
        shared_layer = torch.nn.Linear(16, 3)
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        loss_fn.eval()

        x = torch.randn(8, 16)
        preds = shared_layer(x)
        targets = torch.randn(8, 3)

        initial_weights = loss_fn.weights.clone()
        loss_fn(preds, targets, shared_layer=shared_layer)
        assert torch.allclose(loss_fn.weights, initial_weights)

    def test_per_task_losses_attribute_exists(self):
        """After forward(), _per_task_losses should be set."""
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        _ = loss_fn(preds, targets)
        assert hasattr(loss_fn, "_per_task_losses")

    def test_per_task_losses_shape(self):
        """_per_task_losses shape should match [num_endpoints]."""
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        _ = loss_fn(preds, targets)
        assert loss_fn._per_task_losses.shape == (3,)

    def test_per_task_losses_detached(self):
        """_per_task_losses should not require grad."""
        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        _ = loss_fn(preds, targets)
        assert not loss_fn._per_task_losses.requires_grad
