"""Sklearn-compatible wrappers for graph neural network architectures."""

from matcha.sklearn.graph.attentive_fp import (
    AttentiveFPClassifier,
    AttentiveFPRegressor,
)
from matcha.sklearn.graph.chemprop import ChempropClassifier, ChempropRegressor
from matcha.sklearn.graph.gatedgcn import GatedGCNClassifier, GatedGCNRegressor
from matcha.sklearn.graph.gin import GINClassifier, GINRegressor
from matcha.sklearn.graph.gps import GPSClassifier, GPSRegressor
from matcha.sklearn.graph.gt import GTClassifier, GTRegressor

__all__ = [
    "GINClassifier",
    "GINRegressor",
    "AttentiveFPClassifier",
    "AttentiveFPRegressor",
    "GatedGCNClassifier",
    "GatedGCNRegressor",
    "GPSClassifier",
    "GPSRegressor",
    "ChempropClassifier",
    "ChempropRegressor",
    "GTClassifier",
    "GTRegressor",
]
