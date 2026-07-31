"""Prediction head modules that map encoder representations to task outputs."""

from matcha.torch.predictors.mlp import MLP
from matcha.torch.predictors.snn import SNN

__all__ = ["MLP", "SNN"]
