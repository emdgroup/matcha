"""Self-Normalizing Neural Network (SNN) predictor with TabM BatchEnsemble layers."""

import torch
import torch.nn as nn
from matcha.torch.predictors.base_predictor import BasePredictor, PredictorRegistry


class BatchEnsembleLinear(nn.Module):
    """Parameter-efficient TabM-style BatchEnsemble linear layer.

    Factorizes ``k`` parallel linear layers as a single shared weight
    ``W`` of shape ``(out, in)`` plus per-member rank-1 adapters
    ``R`` of shape ``(k, in)`` and ``S`` of shape ``(k, out)``, and an optional
    per-member bias of shape ``(k, 1, out)``. For a member ``i`` and input row
    ``x``, the layer computes:

    .. math::

        l_i(x) = \\bigl((x \\odot R_i)\\, W^{\\top}\\bigr) \\odot S_i + B_i,

    which reduces the parameter cost of a naive parallel ensemble from
    ``k * in * out`` to ``in * out + k*(in + out) + k*out`` (bias).

    The layer accepts either a 2D input ``(batch, in)`` — internally
    broadcast to ``(k, batch, in)`` — or a 3D input ``(k, batch, in)``.
    Output is always ``(k, batch, out)``.

    Initialization follows the TabM recipe:

    - ``W`` — LeCun-normal (Kaiming-normal with ``mode='fan_in'``, ``nonlinearity='linear'``).
    - ``R`` — random Rademacher ``\\pm 1`` when ``first_layer=True`` to
      diversify the ``k`` submodels at initialization; deterministic ``1``
      otherwise. Without the first-layer Rademacher init, all ``k`` branches
      collapse to identical outputs.
    - ``S`` — deterministic ``1``.
    - ``bias`` — ``0``.

    Paper (TabM): Gorishniy et al. 2024, `arXiv:2410.24210
    <https://arxiv.org/abs/2410.24210>`_.

    :param int in_features: Input feature dimensionality.
    :param int out_features: Output feature dimensionality.
    :param int num_parallel: Number of parallel ensemble members ``k``.
    :param bool first_layer: If True, initialize ``R`` with Rademacher
        ``\\pm 1`` values to diversify the submodels; else initialize to ``1``.
    :param bool bias: Whether to include a per-member bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_parallel: int,
        *,
        first_layer: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_parallel = num_parallel
        self.first_layer = first_layer

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.R = nn.Parameter(torch.empty(num_parallel, in_features))
        self.S = nn.Parameter(torch.empty(num_parallel, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_parallel, 1, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the TabM init recipe to ``W``, ``R``, ``S``, and bias."""
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
        if self.first_layer:
            with torch.no_grad():
                signs = torch.randint(
                    0, 2, self.R.shape, dtype=self.R.dtype, device=self.R.device
                )
                self.R.copy_(signs * 2 - 1)
        else:
            nn.init.ones_(self.R)
        nn.init.ones_(self.S)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def num_extra_parameters(self) -> int:
        """Return the number of per-member (non-shared) parameters."""
        n = self.R.numel() + self.S.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_parallel={self.num_parallel}, first_layer={self.first_layer}, "
            f"bias={self.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the BatchEnsemble forward pass.

        :param torch.Tensor x: Input tensor of shape ``(batch, in_features)``
            or ``(num_parallel, batch, in_features)``.
        :returns: Output tensor of shape ``(num_parallel, batch, out_features)``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x`` is neither 2D nor 3D.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.num_parallel, -1, -1)
        elif x.dim() != 3:
            raise ValueError(
                f"BatchEnsembleLinear expects input of shape (batch, in) or "
                f"(num_parallel, batch, in); got tensor of dim {x.dim()}."
            )
        # x: (k, batch, in); R: (k, in) -> (k, 1, in); S: (k, out) -> (k, 1, out)
        out = (x * self.R.unsqueeze(1)) @ self.weight.t()
        out = out * self.S.unsqueeze(1)
        if self.bias is not None:
            out = out + self.bias
        return out


@PredictorRegistry.register()
class SNN(BasePredictor):
    """Self-Normalizing Neural Network (SNN) predictor with TabM BatchEnsemble.

    Combines the self-normalizing recipe (SELU + AlphaDropout) from Klambauer
    et al. 2017 with the parameter-efficient BatchEnsemble factorization from
    TabM (Gorishniy et al. 2024): each linear layer is a :class:`BatchEnsembleLinear`
    with a shared weight ``W`` and per-member rank-1 adapters ``R``, ``S`` plus
    per-member bias. Outputs from the ``num_parallel`` ensemble members are
    averaged along the leading dimension at the end of :meth:`forward` and
    (for a body with hidden layers) at the end of :meth:`encode`.

    Papers:

    - Klambauer et al. 2017, `arXiv:1706.02515 <https://arxiv.org/abs/1706.02515>`_.
    - Gorishniy et al. 2024, `arXiv:2410.24210 <https://arxiv.org/abs/2410.24210>`_.

    It inherits from :class:`.BasePredictor` for common routines and is intended
    to be used inside a :class:`BaseClassicModel` instance.

    :param int input_dim: input feature dimensionality.
    :param list[int] hidden_dims: shape of hidden layers in the predictor.
        If ``None`` or empty, goes directly from input to output.
    :param int num_endpoints: number of endpoints (if multitasking) or classes
        (if classification) to predict.
    :param float dropout: dropout rate applied between layers (uses AlphaDropout).
    :param int num_parallel: number of parallel ensemble members ``k`` (default: 8).
        Must be ``>= 2``; use the MLP predictor for the single-model case.
    :raises ValueError: If ``num_parallel <= 1``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None,
        num_endpoints: int,
        dropout: float,
        num_parallel: int = 8,
    ):
        super().__init__()

        if num_parallel <= 1:
            raise ValueError(
                f"SNN requires num_parallel >= 2 for ensembling; got {num_parallel}. "
                "Use the MLP predictor for the single-model case."
            )

        self.num_parallel = num_parallel
        hidden_dims = hidden_dims or []
        self._latent_dim: int = hidden_dims[-1] if hidden_dims else input_dim

        dims = [input_dim, *hidden_dims, num_endpoints]

        self.layers = nn.ModuleList()
        self.post = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                BatchEnsembleLinear(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    num_parallel=num_parallel,
                    first_layer=(i == 0),
                )
            )
            is_last = i == len(dims) - 2
            if is_last:
                self.post.append(nn.Identity())
            else:
                self.post.append(nn.Sequential(nn.SELU(), nn.AlphaDropout(p=dropout)))

    def _body(self, x: torch.Tensor, *, include_last: bool) -> torch.Tensor:
        """Run the ensemble body over ``x``.

        :param x: Input tensor of shape ``(batch, input_dim)``.
        :param include_last: If True, run through every layer and post-op.
            If False, stop after the second-to-last layer (used by :meth:`encode`).
        :returns: Tensor of shape ``(num_parallel, batch, feat)`` — or ``x`` unchanged
            when ``include_last=False`` and there are no hidden layers.
        :rtype: torch.Tensor
        """
        stop = len(self.layers) if include_last else len(self.layers) - 1
        for i in range(stop):
            x = self.layers[i](x)
            x = self.post[i](x)
        return x

    def encode(self, mol_features: torch.Tensor) -> torch.Tensor:
        """Extract the latent representation from all layers except the last.

        Averages across the ``num_parallel`` dimension when a hidden body is
        present; otherwise returns the raw input.

        :param mol_features: input tensor of shape ``(batch, input_dim)``.
        :returns: averaged latent representation of shape ``(batch, latent_dim)``.
        :rtype: torch.Tensor
        """
        x = self._body(mol_features, include_last=False)
        if x.dim() == 3:
            x = x.mean(dim=0)
        return x

    def forward(self, mol_features: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass and average across the ``num_parallel`` dimension.

        :param mol_features: input tensor of shape ``(batch, input_dim)``.
        :returns: predictions of shape ``(batch, num_endpoints)``.
        :rtype: torch.Tensor
        """
        x = self._body(mol_features, include_last=True)
        return x.mean(dim=0)
