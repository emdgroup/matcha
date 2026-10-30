"""Tests for matcha.nn.optimizers – OptimizerRegistry."""

import pytest
import torch

from matcha.nn.optimizers import Adam, AdamW, OptimizerRegistry, SGD


# ===================================================================
# Registry completeness
# ===================================================================


class TestOptimizerRegistry:
    @pytest.mark.parametrize(
        "key,expected_class,expected_parent",
        [
            ("adam", Adam, torch.optim.Adam),
            ("adamw", AdamW, torch.optim.AdamW),
            ("sgd", SGD, torch.optim.SGD),
        ],
    )
    def test_alias_resolves_to_upstream_wrapper(
        self, key, expected_class, expected_parent
    ):
        optimizer_class = OptimizerRegistry[key]
        assert optimizer_class is expected_class
        assert issubclass(optimizer_class, expected_parent)
