"""Mean-Variance Estimation predictor head (means + log-variances)."""

import torch
import torch.nn as nn
from matcha.nn.layers import LnBnDr
from matcha.torch.predictors.base_predictor import BasePredictor, PredictorRegistry


@PredictorRegistry.register(alias="mve")
class MVEPredictor(BasePredictor):
    """Predictor head for Mean-Variance Estimation (MVE) uncertainty.

    Reuses the :class:`MLP`-style ``LnBnDr`` hidden body and splits the
    prediction into two parallel :class:`torch.nn.Linear` heads: one producing
    the per-endpoint means and one producing the per-endpoint log-variances.
    Their outputs are stacked along a trailing axis, so :meth:`forward`
    returns a tensor of shape ``(batch, num_endpoints, 2)`` with means at
    ``[..., 0]`` and log-variances at ``[..., 1]``. This matches the layout
    chemprop's ``MveFFN`` produces, so third-party MVE heads can drop into
    matcha without a reshape adapter.

    Predicting the log-variance (rather than the variance) keeps the head
    unconstrained and numerically stable; downstream β-NLL losses
    (:class:`matcha.nn.losses.BetaNLLLoss`,
    :class:`matcha.nn.losses.BoundedBetaNLLLoss`) exponentiate it internally
    and clamp for stability.

    Intended to be used inside a :class:`BaseClassicModel` instance whenever
    ``uncertainty == "mve"``.

    :param int input_dim: Input feature dimensionality.
    :param hidden_dims: Shape of the shared hidden body. ``None`` skips it.
    :type hidden_dims: list[int] or None
    :param int num_endpoints: Number of prediction endpoints. The trailing
        axis of the forward output has size 2 (mean, log-variance).
    :param float dropout: Dropout rate applied inside each ``LnBnDr`` block.
    :param str activation: Activation name resolved via
        :data:`matcha.nn.activations.ActivationRegistry`.
    :param norm: Normalization name resolved via
        :data:`matcha.nn.layers.LayerRegistry`, or ``None``.
    :type norm: str or None
    """

    _returns_mean_and_log_var = True

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None,
        num_endpoints: int,
        dropout: float,
        activation: str,
        norm: str | None,
    ):
        super().__init__()
        self.num_endpoints = num_endpoints
        self._latent_dim: int = hidden_dims[-1] if hidden_dims else input_dim

        if hidden_dims is not None:
            dim_list = [input_dim] + hidden_dims
            layers = []
            for i in range(len(dim_list) - 1):
                layers.append(
                    LnBnDr(dim_list[i], dim_list[i + 1], dropout, activation, norm)
                )
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = None
            dim_list = [input_dim]

        self.mean_head = nn.Linear(dim_list[-1], num_endpoints)
        self.log_var_head = nn.Linear(dim_list[-1], num_endpoints)

    @property
    def prediction_head(self) -> nn.Module:
        """Alias for the mean head so :attr:`BasePredictor.forward` fallbacks
        that inspect ``self.prediction_head`` continue to see a valid module.

        The actual forward pass concatenates means and log-variances via the
        overridden :meth:`forward` below; this attribute is exposed only for
        introspection.
        """
        return self.mean_head

    def forward(self, mol_features: torch.Tensor) -> torch.Tensor:
        """Run the shared body then both heads and stack their outputs.

        :param torch.Tensor mol_features: Input tensor from the encoder of
            shape ``(batch, input_dim)``.
        :returns: Stacked tensor of shape ``(batch, num_endpoints, 2)`` with
            means at ``[..., 0]`` and log-variances at ``[..., 1]``.
        :rtype: torch.Tensor
        """
        if self.layers is not None:
            mol_features = self.encode(mol_features)
        mean = self.mean_head(mol_features)
        log_var = self.log_var_head(mol_features)
        return torch.stack([mean, log_var], dim=-1)
