"""Multi-Layer Perceptron predictor head with optional per-task branches."""

import torch.nn as nn
from matcha.nn.layers import LnBnDr, MultiMLP
from matcha.torch.predictors.base_predictor import BasePredictor, PredictorRegistry


@PredictorRegistry.register()
class MLP(BasePredictor):
    """Multi Layer Perceptron (MLP) utility class.
    It inherits from :class:`.BasePredictor` for common routines (e.g. forward pass).

    It is intended to be used inside a :class:`BaseClassicModel` instance. Check
    the docs of :class:`matcha.torch.models.classic.MLPModel` for further details.

    :param int input_dim: input feature dimensionality

    :param list[int] hidden_dims: shape of shared hidden MLP layers in the predictor
        across all endpoints

    :param list[int] task_head_dims: shape of the MLP layers dedicated to each
        task independently, defaults to None (skipped)

    :param str activation: activation function to use across the network

    :param float dropout: dropout rate between all layers

    :param int num_endpoints: number of endpoints (if multitasking) or classes,
        (if classification) to predict

    :param bool batchnorm: whether to use batchnorm across all layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None,
        task_head_dims: list[int] | None,
        num_endpoints: int,
        dropout: float,
        activation: str,
        norm: str | None,
    ):
        super().__init__()
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
        if task_head_dims is None:
            self.prediction_head = nn.Linear(dim_list[-1], num_endpoints)
        else:
            self.prediction_head = MultiMLP(
                dim_list[-1],
                task_head_dims,
                num_endpoints,
                dropout,
                activation,
                "multibatch",
            )
