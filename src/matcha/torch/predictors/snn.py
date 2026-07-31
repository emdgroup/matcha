"""Self-Normalizing Neural Network (SNN) predictor head with parallel MultiLn layers."""

import torch
import torch.nn as nn
from matcha.torch.predictors.base_predictor import BasePredictor, PredictorRegistry
from matcha.nn.layers import MultiLn


@PredictorRegistry.register()
class SNN(BasePredictor):
    """Self-Normalizing Neural Network (SNN) utility class.

    SNNs use SELU activations and AlphaDropout to maintain self-normalizing properties,
    as described in the paper "Self-Normalizing Neural Networks" (https://arxiv.org/abs/1706.02515).

    This implementation uses MultiLn layers for parallel computation and averages
    the outputs across the num_parallel dimension for ensemble-like behavior.

    It inherits from :class:`.BasePredictor` for common routines (e.g. forward pass).

    It is intended to be used inside a :class:`BaseClassicModel` instance.

    :param int input_dim: input feature dimensionality

    :param list[int] hidden_dims: shape of hidden layers in the predictor.
        If None, goes directly from input to output.

    :param float dropout: dropout rate between all layers (uses AlphaDropout)

    :param int num_endpoints: number of endpoints (if multitasking) or classes
        (if classification) to predict

    :param int num_parallel: number of parallel heads in MultiLn layers (default: 8)
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

        self.num_parallel = num_parallel
        self._latent_dim: int = hidden_dims[-1] if hidden_dims else input_dim

        # SNNs always use SELU activation and AlphaDropout
        activation = nn.SELU()
        alpha_dropout = nn.AlphaDropout(p=dropout)

        # Build layer dimensions
        if hidden_dims is not None and len(hidden_dims) > 0:
            dim_list = [input_dim] + hidden_dims + [num_endpoints]

            # Build layers - store MultiLn layers separately for proper handling
            self.multiln_layers = nn.ModuleList()
            self.activation_layers = nn.ModuleList()
            self.dropout_layers = nn.ModuleList()

            for layer_idx in range(len(dim_list) - 1):
                # Add MultiLn layer
                fc = MultiLn(
                    in_dim=dim_list[layer_idx],
                    out_dim=dim_list[layer_idx + 1],
                    num_parallel=num_parallel,
                )
                self.multiln_layers.append(fc)

                # Add activation and dropout for all layers except the last
                if layer_idx < len(dim_list) - 2:
                    self.activation_layers.append(activation)
                    self.dropout_layers.append(alpha_dropout)

            self.has_hidden = True

        else:
            # No hidden layers - direct connection from input to output
            self.multiln_layers = nn.ModuleList(
                [
                    MultiLn(
                        in_dim=input_dim,
                        out_dim=num_endpoints,
                        num_parallel=num_parallel,
                    )
                ]
            )
            self.activation_layers = nn.ModuleList()
            self.dropout_layers = nn.ModuleList()
            self.has_hidden = False

        # Initialize parameters according to SNN paper
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights using Lecun normal initialization (Kaiming with fan_in mode)
        and biases to zero, as specified in the SNN paper.
        """
        for module in self.modules():
            if isinstance(module, MultiLn):
                # Lecun normal initialization for weights
                # MultiLn weight shape: [num_parallel, out_dim, in_dim]
                for i in range(module.weight.size(0)):
                    nn.init.kaiming_normal_(
                        module.weight[i], mode="fan_in", nonlinearity="linear"
                    )
                # Zero initialization for biases
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode(self, mol_features: torch.Tensor) -> torch.Tensor:
        """Extract the latent representation from all layers except the last.

        Averages across the ``num_parallel`` dimension.

        :param mol_features: input tensor of shape ``(batch, input_dim)``.
        :returns: averaged latent representation of shape ``(batch, hidden_dims[-1])``.
        :rtype: torch.Tensor
        """
        # Apply all layers except the last MultiLn
        x = mol_features
        for i, fc in enumerate(self.multiln_layers[:-1]):
            x = fc(x)  # Output: [num_parallel, batch, features]
            if i < len(self.activation_layers):
                x = self.activation_layers[i](x)
                x = self.dropout_layers[i](x)

        # Average across the num_parallel dimension
        if x.dim() == 3:
            x = x.mean(dim=0)  # [batch, features]

        return x

    def forward(self, mol_features: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass and average across the ``num_parallel`` dimension.

        :param mol_features: input tensor of shape ``(batch, input_dim)``.
        :returns: predictions of shape ``(batch, num_endpoints)``.
        :rtype: torch.Tensor
        """
        x = mol_features
        for i, fc in enumerate(self.multiln_layers):
            x = fc(x)  # Output: [num_parallel, batch, features]
            if i < len(self.activation_layers):
                x = self.activation_layers[i](x)
                x = self.dropout_layers[i](x)

        # Average across the num_parallel dimension
        x = x.mean(dim=0)  # [batch, num_endpoints]

        return x
