"""Base class and registry for predictor heads used in classic models."""

import torch
import torch.nn as nn
from matcha.utils.registry import ClassRegistry


class BasePredictor(nn.Module):
    """Base class for all predictors.

    Not meant to be instantiated directly; subclass this and define
    ``self.layers`` (the hidden body) and ``self.prediction_head``
    (the final linear mapping). The standard :meth:`forward` and
    :meth:`encode` methods then work out of the box.
    """

    def __init__(self):
        super().__init__()

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the representation produced by the hidden layers,
        i.e. the input to the prediction head."""
        return self._latent_dim

    def encode(self, mol_features: torch.Tensor) -> torch.Tensor:
        """Extract the latent representation from the hidden layers.

        :param mol_features: input tensor from the encoder.
        :returns: latent representation tensor.
        :rtype: torch.Tensor
        """
        if self.layers is not None:
            return self.layers(mol_features)
        else:
            return mol_features

    def forward(self, mol_features: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass (hidden layers + prediction head).

        :param mol_features: input tensor from the encoder.
        :returns: predictions tensor.
        :rtype: torch.Tensor
        """
        if self.layers is not None:
            mol_features = self.encode(mol_features)
        return self.prediction_head(mol_features)


PredictorRegistry = ClassRegistry[BasePredictor]()
