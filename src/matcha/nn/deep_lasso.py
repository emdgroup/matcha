"""Deep Lasso input-gradient regularization.

Reference: https://arxiv.org/pdf/2311.05877
"""

import torch.autograd as autograd


def add_dimension_glasso(var, dim=0):
    """Compute the group-lasso norm (L2 per group, then summed) over a dimension.

    :param torch.Tensor var: Gradient tensor.
    :param int dim: Dimension along which to compute the L2 norms.
    :returns: Scalar group-lasso regularization value.
    :rtype: torch.Tensor
    """
    return var.pow(2).sum(dim=dim).add(1e-8).pow(1 / 2.0).sum()


def deep_lasso_regularizer(loss, inputs):
    """Compute the Deep Lasso regularization term.

    Calculates the group-lasso norm of the gradient of *loss* with respect to
    *inputs*, encouraging sparsity in the input-gradient space.

    :param torch.Tensor loss: Scalar loss tensor (must support ``autograd.grad``).
    :param torch.Tensor inputs: Input tensor that ``loss`` was computed from.
    :returns: Scalar regularization value.
    :rtype: torch.Tensor
    """
    grad_params = autograd.grad(loss, inputs, create_graph=True, allow_unused=True)
    regval = add_dimension_glasso(grad_params[0], dim=0)
    return regval
