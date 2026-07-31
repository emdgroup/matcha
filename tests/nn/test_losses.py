"""Tests for matcha.nn.losses – LossRegistry and loss modules."""

import pytest
import torch
import torch.nn.functional as F

from matcha.nn.losses import (
    LossRegistry,
    BCEFocalLoss,
    Poly1BCELoss,
    MultitaskLoss,
    MultiLoss,
    BoundedLoss,
    WeightedBCELoss,
    GradNormLoss,
)


# ===================================================================
# Registry completeness
# ===================================================================


class TestLossRegistryKeys:
    EXPECTED_KEYS = [
        "focal-bce",
        "poly1-bce",
        "multitask",
        "multiloss",
        "bounded",
        "mse",
        "mae",
        "huber",
        "smoothl1",
        "bounded-mse",
        "bounded-mae",
        "bounded-huber",
        "bounded-smoothl1",
        "bce",
        "weighted-bce",
        "gradnorm",
    ]

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_key_registered(self, key):
        assert key in LossRegistry, f"'{key}' not found in LossRegistry"


# ===================================================================
# BCEFocalLoss
# ===================================================================


class TestBCEFocalLoss:
    def test_output_scalar_mean(self):
        loss_fn = BCEFocalLoss(gamma=2, reduction="mean")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar

    def test_output_scalar_sum(self):
        loss_fn = BCEFocalLoss(gamma=2, reduction="sum")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_output_none_reduction(self):
        loss_fn = BCEFocalLoss(gamma=2, reduction="none")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == (8, 1)

    def test_loss_non_negative(self):
        loss_fn = BCEFocalLoss(gamma=2, reduction="mean")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_alpha_weighting(self):
        loss_fn = BCEFocalLoss(gamma=2, alpha=0.75, reduction="mean")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_gamma_zero_matches_bce(self):
        """With gamma=0, focal loss should reduce to BCE (up to alpha)."""
        torch.manual_seed(0)
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16, 1)).float()
        focal = BCEFocalLoss(gamma=0, alpha=None, reduction="mean")(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        assert torch.allclose(focal, bce, atol=1e-5)

    def test_grad_flows(self):
        loss_fn = BCEFocalLoss(gamma=2, reduction="mean")
        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


# ===================================================================
# Poly1BCELoss
# ===================================================================


class TestPoly1BCELoss:
    def test_output_scalar(self):
        loss_fn = Poly1BCELoss(epsilon=1.0, reduction="mean")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_none_reduction(self):
        loss_fn = Poly1BCELoss(epsilon=1.0, reduction="none")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == (8, 1)

    def test_epsilon_zero_matches_bce(self):
        """With epsilon=0, Poly1BCE should reduce to BCE."""
        torch.manual_seed(0)
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16, 1)).float()
        poly = Poly1BCELoss(epsilon=0.0, reduction="mean")(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        assert torch.allclose(poly, bce, atol=1e-5)

    def test_grad_flows(self):
        loss_fn = Poly1BCELoss(reduction="mean")
        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


# ===================================================================
# Simple wrapped losses – only check registry wiring
# ===================================================================


class TestSimpleLosses:
    @pytest.mark.parametrize(
        "key,expected_parent",
        [
            ("mse", torch.nn.MSELoss),
            ("mae", torch.nn.L1Loss),
            ("huber", torch.nn.HuberLoss),
            ("smoothl1", torch.nn.SmoothL1Loss),
            ("bce", torch.nn.BCEWithLogitsLoss),
        ],
    )
    def test_is_subclass(self, key, expected_parent):
        loss_cls = LossRegistry[key]
        assert issubclass(loss_cls, expected_parent)


# ===================================================================
# WeightedBCELoss
# ===================================================================


class TestWeightedBCELoss:
    def test_output_scalar(self):
        loss_fn = WeightedBCELoss(w1=0.7, reduction="mean")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_non_negative(self):
        loss_fn = WeightedBCELoss(w1=0.7, reduction="mean")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_invalid_w1_raises(self):
        with pytest.raises(ValueError):
            WeightedBCELoss(w1=0.0)
        with pytest.raises(ValueError):
            WeightedBCELoss(w1=1.0)
        with pytest.raises(ValueError):
            WeightedBCELoss(w1=-0.5)

    def test_reduction_none(self):
        loss_fn = WeightedBCELoss(w1=0.5, reduction="none")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == (8, 1)

    def test_reduction_sum(self):
        loss_fn = WeightedBCELoss(w1=0.5, reduction="sum")
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_equal_weights_matches_bce(self):
        """w1=0.5 should give 0.5 * BCE since both classes have equal weight."""
        torch.manual_seed(0)
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16, 1)).float()
        weighted = WeightedBCELoss(w1=0.5, reduction="mean")(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        # both weights are 0.5, so weighted loss = 0.5 * bce
        assert torch.allclose(weighted, 0.5 * bce, atol=1e-5)


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

    def test_output_scalar_training(self, loss_configs):
        loss_fn = MultiLoss(loss_configs)
        loss_fn.train()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        result = loss_fn(preds, targets, T_current=0)
        # In training mode, returns only the loss tensor
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

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
        loss = loss_fn(preds, targets, T_current=0)
        assert torch.isfinite(loss)


# ===================================================================
# BoundedLoss
# ===================================================================


class TestBoundedLoss:
    def test_output_scalar(self):
        loss_fn = BoundedLoss(loss_fn="mse")
        preds = torch.randn(8, 1)
        # targets: shape (batch, 1, 2) -- (value, mask)
        targets = torch.zeros(8, 1, 2)
        targets[:, :, 0] = torch.randn(8, 1)  # actual values
        targets[:, :, 1] = 0  # no bound (exact)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0

    def test_lt_bound_no_penalty_when_below(self):
        """With lt_mask (mask=-1), predictions below target should incur no extra penalty."""
        loss_fn = BoundedLoss(loss_fn="mse")
        # target=5.0, mask=-1 (less-than bound)
        targets = torch.tensor([[[5.0, -1.0]]])
        # prediction < target → should be clamped to target
        preds_below = torch.tensor([3.0])
        loss = loss_fn(preds_below, targets)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_gt_bound_no_penalty_when_above(self):
        """With gt_mask (mask=1), predictions above target should incur no extra penalty."""
        loss_fn = BoundedLoss(loss_fn="mse")
        targets = torch.tensor([[[5.0, 1.0]]])
        preds_above = torch.tensor([7.0])
        loss = loss_fn(preds_above, targets)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


# ===================================================================
# BoundedLoss registry aliases
# ===================================================================


class TestBoundedAliases:
    @pytest.mark.parametrize(
        "key", ["bounded-mse", "bounded-mae", "bounded-huber", "bounded-smoothl1"]
    )
    def test_instantiation(self, key):
        loss_fn = LossRegistry[key]()
        assert loss_fn is not None

    @pytest.mark.parametrize(
        "key", ["bounded-mse", "bounded-mae", "bounded-huber", "bounded-smoothl1"]
    )
    def test_forward(self, key):
        loss_fn = LossRegistry[key]()
        preds = torch.randn(8, 1)
        targets = torch.zeros(8, 1, 2)
        targets[:, :, 0] = torch.randn(8, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0


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
