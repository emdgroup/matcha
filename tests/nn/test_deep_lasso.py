"""Tests for matcha.nn.deep_lasso – gradient-based regularisation utilities."""

import torch
import torch.nn as nn

from matcha.nn.deep_lasso import add_dimension_glasso, deep_lasso_regularizer


# ===================================================================
# add_dimension_glasso
# ===================================================================


class TestAddDimensionGlasso:
    def test_returns_scalar(self):
        x = torch.randn(4, 8)
        out = add_dimension_glasso(x, dim=0)
        assert out.dim() == 0

    def test_non_negative(self):
        x = torch.randn(4, 8)
        out = add_dimension_glasso(x, dim=0)
        assert out.item() >= 0

    def test_zero_input_near_zero(self):
        x = torch.zeros(4, 8)
        out = add_dimension_glasso(x, dim=0)
        # Should be close to 0 (small eps contributes sqrt(eps)*cols)
        assert out.item() < 1e-2

    def test_different_dims(self):
        x = torch.randn(4, 8)
        out0 = add_dimension_glasso(x, dim=0)
        out1 = add_dimension_glasso(x, dim=1)
        # Results should generally differ for non-square tensors
        assert out0.shape == out1.shape == ()

    def test_grad_flows(self):
        x = torch.randn(4, 8, requires_grad=True)
        out = add_dimension_glasso(x, dim=0)
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ===================================================================
# deep_lasso_regularizer
# ===================================================================


class TestDeepLassoRegularizer:
    def test_returns_scalar(self):
        model = nn.Linear(8, 4)
        x = torch.randn(2, 8, requires_grad=True)
        y = model(x)
        loss = y.sum()
        reg = deep_lasso_regularizer(loss, x)
        assert reg.dim() == 0

    def test_non_negative(self):
        model = nn.Linear(8, 4)
        x = torch.randn(2, 8, requires_grad=True)
        y = model(x)
        loss = y.sum()
        reg = deep_lasso_regularizer(loss, x)
        assert reg.item() >= 0

    def test_grad_flows_to_model_params(self):
        """The regulariser should allow gradients to flow back to model params."""
        model = nn.Linear(8, 4)
        x = torch.randn(2, 8, requires_grad=True)
        y = model(x)
        loss = y.sum()
        reg = deep_lasso_regularizer(loss, x)
        total = loss + 0.1 * reg
        total.backward()
        # Model weights should have gradients
        assert model.weight.grad is not None

    def test_larger_weights_give_larger_reg(self):
        """Larger model weights should produce larger gradient norms → larger reg."""
        x = torch.randn(4, 8, requires_grad=True)

        # Small weights
        model_small = nn.Linear(8, 4, bias=False)
        nn.init.uniform_(model_small.weight, -0.01, 0.01)
        loss_small = model_small(x).sum()
        reg_small = deep_lasso_regularizer(loss_small, x)

        x2 = x.detach().clone().requires_grad_(True)
        # Large weights
        model_large = nn.Linear(8, 4, bias=False)
        nn.init.uniform_(model_large.weight, -10, 10)
        loss_large = model_large(x2).sum()
        reg_large = deep_lasso_regularizer(loss_large, x2)

        assert reg_large.item() > reg_small.item()
