"""Tests for matcha.nn.activations – ActivationRegistry and activation modules."""

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
    def test_required_aliases_resolve_to_expected_classes(self):
        expected = {
            "geglu": GEGLU,
            "relu": ReLU,
            "leaky_relu": LeakyReLU,
            "prelu": PReLU,
            "selu": SELU,
            "swish": SILU,
            "tanh": Tanh,
            "elu": ELU,
            "mish": Mish,
            "gelu": GELU,
            "sigmoid": Sigmoid,
            "softmax": Softmax,
        }
        for key, cls in expected.items():
            assert ActivationRegistry[key] is cls, (
                f"ActivationRegistry['{key}'] should resolve to {cls.__name__}"
            )


# ===================================================================
# GEGLU specifics
# ===================================================================


class TestGEGLU:
    """GEGLU splits the last dim in half, so input dim must be even."""

    def test_geglu_contract(self):
        """Deterministic contract across 2-D and 3-D inputs.

        Verifies: halved last dim, ``value * gelu(gate)`` formula, finite
        output, and gradient shape after backward.
        """
        torch.manual_seed(0)
        act = ActivationRegistry["geglu"]()
        cases = [
            ((4, 64), (4, 32)),
            ((2, 10, 64), (2, 10, 32)),
        ]
        for in_shape, expected_shape in cases:
            x = torch.randn(*in_shape, requires_grad=True)
            out = act(x)

            assert out.shape == expected_shape, (
                f"input shape {in_shape} should halve last dim to {expected_shape}"
            )

            value, gate = x.chunk(2, dim=-1)
            expected = value * torch.nn.functional.gelu(gate)
            assert torch.allclose(out, expected, atol=1e-6), (
                f"GEGLU({in_shape}) should equal value * gelu(gate)"
            )

            assert torch.isfinite(out).all(), (
                f"GEGLU output for shape {in_shape} must be finite"
            )

            out.sum().backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
