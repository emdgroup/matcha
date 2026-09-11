"""Tests for dropout loss behavior and fixed aliases."""

import pytest
import torch
import torch.nn.functional as F

from matcha.nn.losses import (
    BCEFocalLoss,
    BCELoss,
    DropoutBCELoss,
    DropoutFocalBCELoss,
    DropoutHuberLoss,
    DropoutLoss,
    DropoutMAELoss,
    DropoutMSELoss,
    DropoutPoly1BCELoss,
    DropoutSmoothL1Loss,
    DropoutWeightedBCELoss,
    HuberLoss,
    L1Loss,
    LossRegistry,
    MSELoss,
    MultiLoss,
    MultitaskLoss,
    Poly1BCELoss,
    SmoothL1Loss,
    WeightedBCELoss,
)


# ===================================================================
# DropoutLoss
# ===================================================================


class TestDropoutLoss:
    def test_output_scalar(self):
        loss_fn = LossRegistry["dropout"](loss_fn="mse", dropout=0.5)
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0

    @pytest.mark.parametrize("mode", ["train", "eval"])
    def test_dropout_zero_matches_inner_mean(self, mode):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0)
        getattr(loss_fn, mode)()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        loss = loss_fn(preds, targets)
        expected = F.mse_loss(preds, targets, reduction="mean")
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_eval_mode_deterministic(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.5)
        loss_fn.eval()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        first = loss_fn(preds, targets)
        second = loss_fn(preds, targets)
        assert torch.equal(first, second)

    def test_train_mode_stochastic_unseeded(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.5, seed=None)
        loss_fn.train()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        first = loss_fn(preds, targets)
        second = loss_fn(preds, targets)
        assert not torch.equal(first, second)

    def test_seed_reproducible_across_instances(self):
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        loss_a = DropoutLoss(loss_fn="mse", dropout=0.5, seed=42)
        loss_b = DropoutLoss(loss_fn="mse", dropout=0.5, seed=42)
        loss_a.train()
        loss_b.train()
        seq_a = [loss_a(preds, targets) for _ in range(4)]
        seq_b = [loss_b(preds, targets) for _ in range(4)]
        for a, b in zip(seq_a, seq_b):
            assert torch.equal(a, b)

    def test_seed_advances_across_forwards(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.5, seed=42)
        loss_fn.train()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        first = loss_fn(preds, targets)
        second = loss_fn(preds, targets)
        assert not torch.equal(first, second)

    def test_grad_flows_through_kept_entries(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.5)
        loss_fn.train()
        preds = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8, 3)
        loss = loss_fn(preds, targets)
        loss.backward()
        assert preds.grad is not None
        assert (preds.grad != 0).any()

    def test_nan_composition(self, multitask_targets):
        """At dropout=0 and eval, matches MultitaskLoss (equal per-task counts)."""
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0)
        loss_fn.eval()
        preds = torch.randn_like(multitask_targets)
        loss = loss_fn(preds, multitask_targets.clone())
        assert torch.isfinite(loss)
        multitask = MultitaskLoss(loss_fn="mse")(preds, multitask_targets.clone())
        assert torch.allclose(loss, multitask, atol=1e-6)

    def test_all_nan_returns_finite_zero(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0)
        preds = torch.randn(8, 3)
        targets = torch.full((8, 3), float("nan"))
        loss = loss_fn(preds, targets)
        assert torch.isfinite(loss)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_invalid_dropout_raises(self, bad):
        with pytest.raises(ValueError):
            DropoutLoss(loss_fn="mse", dropout=bad)

    def test_dropout_zero_boundary_ok(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0)
        assert loss_fn is not None

    @pytest.mark.parametrize("alias", ["multitask", "bounded", "gradnorm", "dropout"])
    def test_rejects_nested_wrappers(self, alias):
        with pytest.raises(ValueError):
            DropoutLoss(loss_fn=alias)

    def test_wrapped_by_multitask_loss_constructs(self):
        """Regression: MultitaskLoss injects reduction="none"; DropoutLoss must tolerate it."""
        loss_fn = MultitaskLoss(
            loss_fn="dropout-mse", loss_args={"dropout": 0.5, "seed": 0}
        )
        assert isinstance(loss_fn.loss, DropoutLoss)

    def test_wrapped_by_multi_loss_constructs(self):
        """Regression: MultiLoss injects reduction="none"; DropoutLoss must tolerate it."""
        loss_configs = [
            {
                "loss_fn": "dropout-mse",
                "loss_args": {"dropout": 0.5, "seed": 0},
                "task_map": [0, 1, 2],
                "init_w": 1.0,
                "final_w": 1.0,
                "T": 1,
                "warmup": 0,
            },
        ]
        loss_fn = MultiLoss(loss_configs)
        assert isinstance(loss_fn.losses[0], DropoutLoss)

    def test_reduction_mean_matches_current_behavior(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0)
        loss_fn.eval()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        loss = loss_fn(preds, targets)
        expected = F.mse_loss(preds, targets, reduction="mean")
        assert loss.dim() == 0
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_reduction_none_returns_per_element(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0, reduction="none")
        loss_fn.eval()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        loss = loss_fn(preds, targets)
        expected = F.mse_loss(preds, targets, reduction="none")
        assert loss.shape == (8, 3)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_reduction_sum_matches_masked_sum(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0, reduction="sum")
        loss_fn.eval()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        loss = loss_fn(preds, targets)
        expected = F.mse_loss(preds, targets, reduction="sum")
        assert loss.dim() == 0
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_reduction_none_zeros_nan_and_dropout(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.5, seed=0, reduction="none")
        loss_fn.train()
        preds = torch.randn(8, 3)
        targets = torch.randn(8, 3)
        targets[0, 1] = float("nan")
        targets[3, 0] = float("nan")

        # Reproduce the internal mask trajectory to identify dropped positions.
        gen = torch.Generator()
        gen.manual_seed(0)
        keep_from_dropout = torch.rand((8, 3), generator=gen) >= 0.5
        nan_mask = torch.isnan(targets)
        keep_mask = (~nan_mask) & keep_from_dropout

        out = loss_fn(preds, targets)
        assert out.shape == (8, 3)
        # NaN and dropped positions must be exactly zero.
        assert torch.all(out[~keep_mask] == 0.0)
        # Kept positions must be non-zero for generic inputs.
        assert torch.all(out[keep_mask] != 0.0)

    def test_reduction_none_preserves_shape(self):
        loss_fn = DropoutLoss(loss_fn="mse", dropout=0.0, reduction="none")
        loss_fn.eval()
        preds = torch.randn(16, 5)
        targets = torch.randn(16, 5)
        out = loss_fn(preds, targets)
        assert out.shape == preds.shape

    def test_wrapped_by_multi_loss_forward(self):
        """End-to-end regression: ``MultiLoss`` with mixed dropout losses runs forward.

        Mirrors the ``pretrain_multitask`` config family from issue #59: one
        entry using ``dropout-mse`` on descriptor columns and one using
        ``dropout-focal-bce`` on fingerprint columns, with sprinkled NaN targets.
        """
        loss_configs = [
            {
                "loss_fn": "dropout-mse",
                "loss_args": {"dropout": 0.3, "seed": 0},
                "task_map": [0, 1, 2],
                "init_w": 1.0,
                "final_w": 1.0,
                "T": 1,
                "warmup": 0,
                "name": "descriptors",
            },
            {
                "loss_fn": "dropout-focal-bce",
                "loss_args": {"dropout": 0.3, "seed": 1},
                "task_map": [3, 4],
                "init_w": 1.0,
                "final_w": 1.0,
                "T": 1,
                "warmup": 0,
                "name": "fingerprints",
            },
        ]
        loss_fn = MultiLoss(loss_configs)
        loss_fn.train()

        preds = torch.randn(8, 5)
        targets = torch.randn(8, 5)
        # Sprinkle NaNs across both branches.
        targets[0, 1] = float("nan")
        targets[5, 2] = float("nan")
        # Binarize the fingerprint columns and add NaNs.
        targets[:, 3:] = (torch.randn(8, 2) > 0.0).float()
        targets[2, 3] = float("nan")

        total_loss, loss_log = loss_fn(preds, targets, T_current=0)
        assert total_loss.dim() == 0
        assert torch.isfinite(total_loss)
        assert "descriptors" in loss_log
        assert "fingerprints" in loss_log

    def test_wrapped_by_multitask_loss_forward(self, multitask_targets):
        """End-to-end regression: ``MultitaskLoss`` wrapping ``dropout-mse`` runs forward."""
        loss_fn = MultitaskLoss(
            loss_fn="dropout-mse", loss_args={"dropout": 0.5, "seed": 0}
        )
        loss_fn.train()
        preds = torch.randn_like(multitask_targets)
        loss = loss_fn(preds, multitask_targets.clone())
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ===================================================================
# DropoutLoss registry aliases
# ===================================================================


class TestDropoutAliases:
    @pytest.mark.parametrize(
        "key,expected_class,expected_inner",
        [
            ("dropout-mse", DropoutMSELoss, MSELoss),
            ("dropout-mae", DropoutMAELoss, L1Loss),
            ("dropout-huber", DropoutHuberLoss, HuberLoss),
            ("dropout-smoothl1", DropoutSmoothL1Loss, SmoothL1Loss),
            ("dropout-bce", DropoutBCELoss, BCELoss),
            ("dropout-focal-bce", DropoutFocalBCELoss, BCEFocalLoss),
            ("dropout-poly1-bce", DropoutPoly1BCELoss, Poly1BCELoss),
            ("dropout-weighted-bce", DropoutWeightedBCELoss, WeightedBCELoss),
        ],
    )
    def test_alias_selection_and_wrapper_keyword_forwarding(
        self, key, expected_class, expected_inner
    ):
        assert LossRegistry[key] is expected_class

        loss_fn = LossRegistry[key](dropout=0.25, seed=7, reduction="sum")

        assert type(loss_fn) is expected_class
        assert type(loss_fn.loss) is expected_inner
        assert loss_fn.dropout == 0.25
        assert loss_fn._seed == 7
        assert loss_fn.reduction == "sum"
        assert loss_fn.loss.reduction == "none"

    @pytest.mark.parametrize(
        "key,kwargs,attribute,expected",
        [
            ("dropout-huber", {"delta": 0.75}, "delta", 0.75),
            ("dropout-smoothl1", {"beta": 0.25}, "beta", 0.25),
            ("dropout-focal-bce", {"gamma": 3}, "gamma", 3),
            ("dropout-poly1-bce", {"epsilon": 2.0}, "epsilon", 2.0),
            ("dropout-weighted-bce", {"w1": 0.7}, "w1", 0.7),
        ],
    )
    def test_alias_forwards_inner_keyword(
        self, key, kwargs, attribute, expected
    ):
        loss_fn = LossRegistry[key](**kwargs)

        assert getattr(loss_fn.loss, attribute) == expected

    def test_bce_alias_forwards_pos_weight(self):
        pos_weight = torch.tensor([2.0])

        loss_fn = LossRegistry["dropout-bce"](pos_weight=pos_weight)

        assert torch.equal(loss_fn.loss.pos_weight, pos_weight)
