"""Tests for cross-schema validators on :class:`ScikitLearnInputModel`.

Covers the MVE / scaler pairing rule enforced at the composed sklearn schema
boundary: ``model.uncertainty == "mve"`` requires
``datamodule.scaler_type == "standard"``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from matcha.utils.schemas.sklearn_api import ScikitLearnInputModel


def _make_config(*, uncertainty: str, scaler_type: str) -> dict[str, Any]:
    """Build a minimal valid MLP + tabular config with the given knobs."""
    if uncertainty == "mve":
        loss_fn = "mve"
    else:
        loss_fn = "mse"

    return {
        "training": {},
        "datamodule": {
            "datamodule_type": "tabular",
            "input_dim": 8,
            "feature_list": ["f0", "f1"],
            "scaler_type": scaler_type,
        },
        "model": {
            "torch_type": "mlp",
            "additional_mol_features_dim": 0,
            "num_endpoints": 1,
            "loss_fn": loss_fn,
            "loss_args": {},
            "optimizer": "adam",
            "optimizer_args": {"lr": 1e-3},
            "scheduler": "cosine_annealing",
            "scheduler_args": {"min_lr": 1e-6, "total_steps": 50},
            "uncertainty": uncertainty,
            "deep_lasso_weight": 0.0,
            "dropout": 0.0,
            "hidden_dims": [16],
            "activation": "relu",
            "task_head_dims": None,
        },
        "metadata": {
            "model_type": "MLPRegressor",
            "model_version": 1,
            "model_name": "test",
            "model_owner": "test",
            "model_scope": "test",
            "matcha_version": "0.0.0",
            "date": "2026-01-01",
            "description": "test",
        },
        "task_type": "regression",
        "calibration": None,
        "mlflow": None,
        "tuning": None,
    }


class TestScikitLearnInputModelMVEScalerPairing:
    """The composed schema rejects MVE with a non-``standard`` scaler."""

    def test_mve_with_standard_scaler_passes(self):
        cfg = _make_config(uncertainty="mve", scaler_type="standard")
        m = ScikitLearnInputModel(**cfg)
        assert m.model.uncertainty == "mve"
        assert m.datamodule.scaler_type == "standard"

    def test_mve_with_quantile_scaler_rejected(self):
        cfg = _make_config(uncertainty="mve", scaler_type="quantile")
        with pytest.raises(ValidationError, match=r"scaler_type='standard'"):
            ScikitLearnInputModel(**cfg)

    def test_mc_dropout_with_quantile_scaler_allowed(self):
        # The scaler constraint is scoped to MVE — MC dropout must not trip it.
        cfg = _make_config(uncertainty="mc-dropout", scaler_type="quantile")
        m = ScikitLearnInputModel(**cfg)
        assert m.model.uncertainty == "mc-dropout"
        assert m.datamodule.scaler_type == "quantile"
