"""Scikit-learn-compatible API for molecular property prediction models."""

# Base classes and utilities
from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry
from matcha.sklearn.ensemble import Ensemble
from matcha.sklearn.finetuner import FinetuningClassifier, FinetuningRegressor
from matcha.sklearn.autoload import autoload

# CLM models
from matcha.sklearn.clm import (
    RoFormerClassifier,
    RoFormerRegressor,
    CNNClassifier,
    CNNRegressor,
    RNNClassifier,
    RNNRegressor,
)

# Graph models
from matcha.sklearn.graph import (
    AttentiveFPClassifier,
    AttentiveFPRegressor,
    ChempropClassifier,
    ChempropRegressor,
    GatedGCNClassifier,
    GatedGCNRegressor,
    GINClassifier,
    GINRegressor,
    GPSClassifier,
    GPSRegressor,
    GTClassifier,
    GTRegressor,
)

# Graph3D models
from matcha.sklearn.graph3d import (
    E3GNNClassifier,
    E3GNNRegressor,
    GPS3DClassifier,
    GPS3DRegressor,
    GT3DClassifier,
    GT3DRegressor,
)

# Tabular models
from matcha.sklearn.tabular import (
    MLPClassifier,
    MLPRegressor,
    SNNClassifier,
    SNNRegressor,
)

__all__ = [
    # Base classes and utilities
    "ScikitLearnModelRegistry",
    "Ensemble",
    "FinetuningClassifier",
    "FinetuningRegressor",
    "autoload",
    # CLM models
    "RoFormerClassifier",
    "RoFormerRegressor",
    "CNNClassifier",
    "CNNRegressor",
    "RNNClassifier",
    "RNNRegressor",
    # Graph models
    "AttentiveFPClassifier",
    "AttentiveFPRegressor",
    "ChempropClassifier",
    "ChempropRegressor",
    "GatedGCNClassifier",
    "GatedGCNRegressor",
    "GINClassifier",
    "GINRegressor",
    "GPSClassifier",
    "GPSRegressor",
    "GTClassifier",
    "GTRegressor",
    # Graph3D models
    "E3GNNClassifier",
    "E3GNNRegressor",
    "GPS3DClassifier",
    "GPS3DRegressor",
    "GT3DClassifier",
    "GT3DRegressor",
    # Tabular models
    "MLPClassifier",
    "MLPRegressor",
    "SNNClassifier",
    "SNNRegressor",
]
