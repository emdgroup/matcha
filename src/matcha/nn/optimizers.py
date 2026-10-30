"""Optimizer wrappers registered in the :data:`OptimizerRegistry`."""

from torch import optim
from matcha.utils.registry import ClassRegistry

OptimizerRegistry = ClassRegistry()


@OptimizerRegistry.register("adam")
class Adam(optim.Adam):
    """Adam optimizer (wraps :class:`torch.optim.Adam`)."""

    pass


@OptimizerRegistry.register("adamw")
class AdamW(optim.AdamW):
    """AdamW optimizer with decoupled weight decay (wraps :class:`torch.optim.AdamW`)."""

    pass


@OptimizerRegistry.register("sgd")
class SGD(optim.SGD):
    """Stochastic gradient descent optimizer (wraps :class:`torch.optim.SGD`)."""

    pass
