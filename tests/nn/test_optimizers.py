"""Tests for matcha.nn.optimizers – OptimizerRegistry."""

import pytest
import torch

from matcha.nn.optimizers import OptimizerRegistry


# ===================================================================
# Registry completeness
# ===================================================================


class TestOptimizerRegistryKeys:
    EXPECTED_KEYS = ["adam", "adamw", "sgd"]

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_key_registered(self, key):
        assert key in OptimizerRegistry, f"'{key}' not found in OptimizerRegistry"


# ===================================================================
# Instantiation
# ===================================================================


class TestOptimizerInstantiation:
    @pytest.fixture()
    def simple_model(self):
        return torch.nn.Linear(10, 2)

    @pytest.mark.parametrize("key", ["adam", "adamw", "sgd"])
    def test_creates_optimizer(self, key, simple_model):
        opt_cls = OptimizerRegistry[key]
        opt = opt_cls(simple_model.parameters(), lr=1e-3)
        assert opt is not None

    @pytest.mark.parametrize("key", ["adam", "adamw", "sgd"])
    def test_has_param_groups(self, key, simple_model):
        opt_cls = OptimizerRegistry[key]
        opt = opt_cls(simple_model.parameters(), lr=0.01)
        assert len(opt.param_groups) > 0


# ===================================================================
# Registry resolves to correct parent class
# ===================================================================


class TestOptimizerParentClass:
    @pytest.mark.parametrize(
        "key,expected_parent",
        [
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
            ("sgd", torch.optim.SGD),
        ],
    )
    def test_is_subclass_of_torch_optimizer(self, key, expected_parent):
        opt_cls = OptimizerRegistry[key]
        assert issubclass(opt_cls, expected_parent)
