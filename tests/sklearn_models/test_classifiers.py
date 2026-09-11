"""Tests for all sklearn *classifier* architectures: fit, predict, predict_proba.

Each test is parametrized over every classifier class (CLM, graph, graph3D,
tabular) so that a failure clearly identifies which architecture broke.

All model classes, kwargs, and parametrized fixtures are defined in conftest.py
and auto-discovered by pytest – no explicit import needed.
"""

from tests.sklearn_models._estimator_contracts import ClassifierContract


class TestClassifier(ClassifierContract):
    pass
