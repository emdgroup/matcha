"""Tests for matcha.nn.activations – ActivationRegistry and activation modules."""

import pytest
import torch

from matcha.nn.activations import ActivationRegistry


# ===================================================================
# Registry completeness
# ===================================================================


class TestActivationRegistryKeys:
    """Ensure every expected key is present in the registry."""

    EXPECTED_KEYS = [
        "geglu",
        "relu",
        "leaky_relu",
        "prelu",
        "selu",
        "swish",
        "tanh",
        "elu",
        "mish",
        "gelu",
        "sigmoid",
        "softmax",
    ]

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_key_registered(self, key):
        assert key in ActivationRegistry, f"'{key}' not found in ActivationRegistry"

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_instantiation(self, key):
        act = ActivationRegistry[key]()
        assert act is not None


# ===================================================================
# Forward pass smoke tests
# ===================================================================


class TestActivationForward:
    """Smoke-test that the registry wrappers can be called through the registry."""

    SIMPLE_KEYS = [
        "relu",
        "leaky_relu",
        "prelu",
        "selu",
        "swish",
        "tanh",
        "elu",
        "mish",
        "gelu",
        "sigmoid",
    ]

    @pytest.mark.parametrize("key", SIMPLE_KEYS)
    def test_callable_through_registry(self, key):
        x = torch.randn(8, 32)
        act = ActivationRegistry[key]()
        out = act(x)
        assert out.shape == x.shape


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


# ===================================================================
# Softmax – registry instantiation with args
# ===================================================================


class TestSoftmax:
    """Softmax wrapper should accept dim arg through the registry."""

    def test_instantiation_with_dim_arg(self):
        act = ActivationRegistry["softmax"](dim=-1)
        x = torch.randn(4, 10)
        out = act(x)
        assert out.shape == x.shape
