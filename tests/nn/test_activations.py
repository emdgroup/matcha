"""Tests for matcha.nn.activations – ActivationRegistry and activation modules."""

import pytest
import torch

from matcha.nn.activations import (
    ActivationRegistry,
    ELU,
    GEGLU,
    GELU,
    LeakyReLU,
    Mish,
    PReLU,
    ReLU,
    SELU,
    SILU,
    Sigmoid,
    Softmax,
    Tanh,
)


# ===================================================================
# Registry completeness
# ===================================================================


class TestActivationRegistry:
    @pytest.mark.parametrize(
        "key,expected_class",
        [
            ("geglu", GEGLU),
            ("relu", ReLU),
            ("leaky_relu", LeakyReLU),
            ("prelu", PReLU),
            ("selu", SELU),
            ("swish", SILU),
            ("tanh", Tanh),
            ("elu", ELU),
            ("mish", Mish),
            ("gelu", GELU),
            ("sigmoid", Sigmoid),
            ("softmax", Softmax),
        ],
    )
    def test_alias_resolves_to_registered_class(self, key, expected_class):
        assert ActivationRegistry[key] is expected_class


# ===================================================================
# GEGLU specifics
# ===================================================================


class TestGEGLU:
    """GEGLU splits the last dim in half, so input dim must be even."""

    def test_output_shape_halves_last_dim(self):
        act = ActivationRegistry["geglu"]()
        x = torch.randn(4, 64)
        out = act(x)
        assert out.shape == (4, 32)

    def test_output_shape_3d(self):
        act = ActivationRegistry["geglu"]()
        x = torch.randn(2, 10, 64)
        out = act(x)
        assert out.shape == (2, 10, 32)

    def test_output_is_finite(self):
        act = ActivationRegistry["geglu"]()
        x = torch.randn(4, 64)
        out = act(x)
        assert torch.isfinite(out).all()

    def test_grad_flows(self):
        act = ActivationRegistry["geglu"]()
        x = torch.randn(4, 64, requires_grad=True)
        out = act(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
