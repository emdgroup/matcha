"""Tests for matcha.nn.optimizers – OptimizerRegistry."""

from matcha.nn.optimizers import Adam, AdamW, OptimizerRegistry, SGD


# ===================================================================
# Registry completeness
# ===================================================================


class TestOptimizerRegistry:
    def test_required_aliases_resolve_to_expected_classes(self):
        expected = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
        }
        for key, cls in expected.items():
            assert OptimizerRegistry[key] is cls, (
                f"OptimizerRegistry['{key}'] should resolve to {cls.__name__}"
            )
