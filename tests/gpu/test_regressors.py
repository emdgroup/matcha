"""GPU tests for sklearn *regressor* architectures: fit and predict on toy data.

Each test is parametrized over a subset of regressor classes
(Chemprop, RoFormer, GIN, SNN) so that a failure clearly identifies
which architecture broke.

All model classes, kwargs, and parametrized fixtures are defined in conftest.py
and auto-discovered by pytest – no explicit import needed.
"""

from tests.sklearn_models._estimator_contracts import RegressorContract


class TestRegressor(RegressorContract):
    pass
