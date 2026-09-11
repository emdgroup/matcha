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
    def test_reduction_variants(self):
        """mean/sum/none reductions produce the expected output shapes."""
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        expected_shapes = {
            "mean": (),
            "sum": (),
            "none": (8, 1),
        }
        for reduction, expected in expected_shapes.items():
            loss = BCEFocalLoss(gamma=2, reduction=reduction)(logits, targets)
            assert loss.shape == expected, (
                f"BCEFocalLoss(reduction={reduction!r}) should produce shape {expected}"
            )

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
    def test_reduction_variants(self):
        """mean and none reductions produce the expected output shapes."""
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        expected_shapes = {
            "mean": (),
            "none": (8, 1),
        }
        for reduction, expected in expected_shapes.items():
            loss = Poly1BCELoss(epsilon=1.0, reduction=reduction)(logits, targets)
            assert loss.shape == expected, (
                f"Poly1BCELoss(reduction={reduction!r}) should produce shape {expected}"
            )

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
    def test_reduction_variants(self):
        """mean/sum/none reductions produce the expected output shapes."""
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        expected_shapes = {
            "mean": (),
            "sum": (),
            "none": (8, 1),
        }
        for reduction, expected in expected_shapes.items():
            loss = WeightedBCELoss(w1=0.5, reduction=reduction)(logits, targets)
            assert loss.shape == expected, (
                f"WeightedBCELoss(reduction={reduction!r}) should produce shape {expected}"
            )

    def test_invalid_w1_raises(self):
        for w1 in (0.0, 1.0, -0.5):
            with pytest.raises(ValueError):
                WeightedBCELoss(w1=w1)

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
    def test_exact_and_bounded_directions(self):
        """Table-driven contract for exact, less-than, and greater-than bounds.

        Preserves both dimensional branches -- batched ``(8, 1)`` preds paired
        with ``(8, 1, 2)`` targets (exact) and single-element preds paired with
        ``(1, 1, 2)`` targets (bounded) -- and both bound directions (lt/gt).
        """
        loss_fn = BoundedLoss(loss_fn="mse")

        # (case name, preds, targets, expected assertion)
        # mask semantics: 0 = exact, -1 = less-than bound, 1 = greater-than bound
        batched_targets = torch.zeros(8, 1, 2)
        batched_targets[:, :, 0] = torch.randn(8, 1)
        # mask defaults to 0 (exact)

        cases = [
            (
                "exact-batched-scalar",
                torch.randn(8, 1),
                batched_targets,
                "scalar",
            ),
            (
                "lt-below-no-penalty",
                torch.tensor([3.0]),
                torch.tensor([[[5.0, -1.0]]]),
                "zero",
            ),
            (
                "gt-above-no-penalty",
                torch.tensor([7.0]),
                torch.tensor([[[5.0, 1.0]]]),
                "zero",
            ),
        ]
        for name, preds, targets, kind in cases:
            loss = loss_fn(preds, targets)
            if kind == "scalar":
                assert loss.dim() == 0, f"{name}: expected scalar output"
            else:
                assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
                    f"{name}: bounded loss should be zero when prediction is within bound"
                )


# ===================================================================
# BoundedLoss registry aliases
# ===================================================================


class TestBoundedAliases:
    def test_alias_selection_and_keyword_forwarding(self):
        """Every bounded alias selects the expected wrapper and inner loss.

        Also verifies that constructor keyword arguments (``reduction='sum'``)
        are forwarded to the inner loss.
        """
        cases = [
            ("bounded-mse", BoundedMSELoss, MSELoss),
            ("bounded-mae", BoundedMAELoss, L1Loss),
            ("bounded-huber", BoundedHuberLoss, HuberLoss),
            ("bounded-smoothl1", BoundedSmoothL1Loss, SmoothL1Loss),
        ]
        for key, expected_class, expected_inner in cases:
            assert LossRegistry[key] is expected_class, (
                f"LossRegistry['{key}'] should resolve to {expected_class.__name__}"
            )

            loss_fn = LossRegistry[key](reduction="sum")

            assert type(loss_fn) is expected_class, (
                f"LossRegistry['{key}'](...) instance type mismatch"
            )
            assert type(loss_fn.loss) is expected_inner, (
                f"LossRegistry['{key}'] inner loss should be {expected_inner.__name__}"
            )
            assert loss_fn.loss.reduction == "sum", (
                f"LossRegistry['{key}'] should forward reduction='sum' to inner loss"
            )
