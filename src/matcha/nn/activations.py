"""Activation functions registered in the :data:`ActivationRegistry`."""

import torch
from torch import nn
import torch.nn.functional as F
from matcha.utils.registry import ClassRegistry

ActivationRegistry = ClassRegistry()


@ActivationRegistry.register("geglu")
class GEGLU(nn.Module):
    """GEGLU activation function.

    Splits the input along the last dimension into a value half and a gate half,
    then returns ``value * GELU(gate)``.

    Reference: https://arxiv.org/abs/2002.05202v1
    """

    def forward(self, x) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor whose last dimension is split in two.
        :returns: Gated output with half the last-dimension size.
        :rtype: torch.Tensor
        """
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


@ActivationRegistry.register("relu")
class ReLU(nn.ReLU):
    """Rectified Linear Unit (wraps :class:`torch.nn.ReLU`)."""

    pass


@ActivationRegistry.register("leaky_relu")
class LeakyReLU(nn.LeakyReLU):
    """Leaky ReLU (wraps :class:`torch.nn.LeakyReLU`)."""

    pass


@ActivationRegistry.register("prelu")
class PReLU(nn.PReLU):
    """Parametric ReLU (wraps :class:`torch.nn.PReLU`)."""

    pass


@ActivationRegistry.register("selu")
class SELU(nn.SELU):
    """Scaled Exponential Linear Unit (wraps :class:`torch.nn.SELU`)."""

    pass


@ActivationRegistry.register("swish")
class SILU(nn.SiLU):
    """SiLU / Swish activation (wraps :class:`torch.nn.SiLU`)."""

    pass


@ActivationRegistry.register("tanh")
class Tanh(nn.Tanh):
    """Hyperbolic tangent (wraps :class:`torch.nn.Tanh`)."""

    pass


@ActivationRegistry.register("elu")
class ELU(nn.ELU):
    """Exponential Linear Unit (wraps :class:`torch.nn.ELU`)."""

    pass


@ActivationRegistry.register("mish")
class Mish(nn.Mish):
    """Mish activation (wraps :class:`torch.nn.Mish`)."""

    pass


@ActivationRegistry.register("gelu")
class GELU(nn.GELU):
    """Gaussian Error Linear Unit (wraps :class:`torch.nn.GELU`)."""

    pass


@ActivationRegistry.register("sigmoid")
class Sigmoid(nn.Sigmoid):
    """Sigmoid activation (wraps :class:`torch.nn.Sigmoid`)."""

    pass


@ActivationRegistry.register("softmax")
class Softmax(nn.Softmax):
    """Softmax activation (wraps :class:`torch.nn.Softmax`)."""

    pass
