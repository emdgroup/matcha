"""Tests for loss aliases and custom single-task loss modules."""

import pytest
import torch
import torch.nn.functional as F

from matcha.nn.losses import (
    LossRegistry,
    BCELoss,
    BCEFocalLoss,
    BoundedHuberLoss,
    BoundedLoss,
    BoundedMAELoss,
    BoundedMSELoss,
    BoundedSmoothL1Loss,
    CrossEntropyLoss,
    HuberLoss,
    L1Loss,
    MSELoss,
    Poly1BCELoss,
    SmoothL1Loss,
    WeightedBCELoss,
)


# ===================================================================
# Registry completeness
# ===================================================================


class TestLossRegistry:
    """Required aliases and exact mappings for inheritance-only losses.

    `docs/source/contributing/adding-a-loss.md` names ``LossRegistry`` as the
    extension point, so every listed alias must remain resolvable. Additional
    registry entries are permitted; only the required subset is enforced.
    """

    REQUIRED_KEYS = [
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
        "cross_entropy",
        "weighted-bce",
        "gradnorm",
        "dropout",
        "dropout-mse",
        "dropout-mae",
        "dropout-huber",
        "dropout-smoothl1",
        "dropout-bce",
        "dropout-focal-bce",
        "dropout-poly1-bce",
        "dropout-weighted-bce",
        "beta-nll",
        "mve",
        "bounded-beta-nll",
    ]

    def test_required_aliases_resolve_to_expected_classes(self):
        for key in self.REQUIRED_KEYS:
            assert key in LossRegistry, f"'{key}' not found in LossRegistry"

        exact_mappings = {
            "mse": MSELoss,
            "mae": L1Loss,
            "huber": HuberLoss,
            "smoothl1": SmoothL1Loss,
            "bce": BCELoss,
            "cross_entropy": CrossEntropyLoss,
        }
        for key, cls in exact_mappings.items():
            assert LossRegistry[key] is cls, (
                f"LossRegistry['{key}'] should resolve to {cls.__name__}"
            )


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
        "key,expected_class,expected_inner",
        [
            ("bounded-mse", BoundedMSELoss, MSELoss),
            ("bounded-mae", BoundedMAELoss, L1Loss),
            ("bounded-huber", BoundedHuberLoss, HuberLoss),
            ("bounded-smoothl1", BoundedSmoothL1Loss, SmoothL1Loss),
        ],
    )
    def test_alias_selection_and_keyword_forwarding(
        self, key, expected_class, expected_inner
    ):
        assert LossRegistry[key] is expected_class

        loss_fn = LossRegistry[key](reduction="sum")

        assert type(loss_fn) is expected_class
        assert type(loss_fn.loss) is expected_inner
        assert loss_fn.loss.reduction == "sum"
