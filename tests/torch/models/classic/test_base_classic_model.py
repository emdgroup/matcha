"""Unit tests for the shared plumbing on :class:`BaseClassicModel`.

Stage 2 of issue #95 introduces the MVE dispatch path through
``_parse_predictor``: a class-level ``_predictor_cls`` default, a
``_predictor_kwargs`` hook, ``predict_variance_step`` for the intrinsic
uncertainty read-out, and a MC-dropout-independent slice in ``predict_step``.
The tests here exercise this plumbing through a minimal toy model that
subclasses :class:`BaseClassicModel` directly (no encoder, tabular batch).
"""

from typing import Any

import pytest
import torch
from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.nn.losses import (
    BetaNLLLoss,
    BoundedBetaNLLLoss,
    MultitaskLoss,
)
from matcha.torch.models.classic.base_classic_model import BaseClassicModel
from matcha.torch.predictors.mlp import MLP
from matcha.torch.predictors.mve import MVEPredictor


class _ToyClassicModel(BaseClassicModel, HyperparametersMixin):
    """Minimal tabular classic model used to exercise the base plumbing.

    Skips the encoder entirely (predictor input dimension is driven by
    ``additional_mol_features_dim``), reuses the ``pred_*`` hparam naming
    convention that :meth:`BaseClassicModel._predictor_kwargs` reads.
    """

    def __init__(
        self,
        additional_mol_features_dim: int = 8,
        pred_hidden_dims: list[int] | None = [16],
        pred_task_head_dims: list[int] | None = None,
        pred_activation: str = "relu",
        pred_dropout: float = 0.0,
        num_endpoints: int = 1,
        uncertainty: str = "mc-dropout",
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 1e-3},
        scheduler: str = "constant",
        scheduler_args: dict = {},
    ):
        super().__init__(additional_mol_features_dim=additional_mol_features_dim)
        self.save_hyperparameters()
        self._parse_predictor()
        self._parse_train_config()


def _make_batch(batch: int, input_dim: int, num_endpoints: int) -> dict[str, Any]:
    return {
        "mol_features": torch.randn(batch, input_dim),
        "y": torch.randn(batch, num_endpoints),
    }


class TestUncertaintyMethodProperty:
    """``uncertainty_method`` is the single canonical source for uncertainty
    dispatch across the codebase — replaces the retired ``_mve_active``
    boolean side-effect flag."""

    def test_default_construction_reports_mc_dropout(self):
        model = _ToyClassicModel(num_endpoints=2)
        assert model.uncertainty_method == "mc-dropout"

    def test_mve_construction_reports_mve(self):
        model = _ToyClassicModel(num_endpoints=2, uncertainty="mve", loss_fn="beta-nll")
        assert model.uncertainty_method == "mve"

    def test_falls_back_to_mc_dropout_when_hparam_missing(self):
        """Defensive path for raw checkpoints saved before the enum landed —
        an absent ``uncertainty`` key must still be interpretable."""
        model = _ToyClassicModel(num_endpoints=2)
        # Simulate the legacy checkpoint case by dropping the hparam entirely.
        del model.hparams["uncertainty"]
        assert model.uncertainty_method == "mc-dropout"

    def test_falls_back_to_mc_dropout_when_hparam_is_none(self):
        """Legacy YAML configs with ``uncertainty: null`` may still show up in
        raw hparam dicts on old checkpoints — the property normalizes them."""
        model = _ToyClassicModel(num_endpoints=2)
        model.hparams["uncertainty"] = None
        assert model.uncertainty_method == "mc-dropout"


class TestParsePredictorDispatch:
    def test_default_uses_predictor_cls_attribute(self):
        model = _ToyClassicModel(num_endpoints=2)
        assert isinstance(model.predictor, MLP)
        assert model.uncertainty_method == "mc-dropout"

    def test_uncertainty_mve_swaps_to_mve_predictor(self):
        model = _ToyClassicModel(num_endpoints=3, uncertainty="mve", loss_fn="beta-nll")
        assert isinstance(model.predictor, MVEPredictor)
        assert model.uncertainty_method == "mve"

    def test_mve_predictor_output_width_is_double(self):
        model = _ToyClassicModel(num_endpoints=3, uncertainty="mve", loss_fn="beta-nll")
        model.eval()
        out = model.predictor(torch.randn(4, 8))
        assert out.shape == (4, 6)

    def test_kwargs_filtered_against_target_signature(self):
        """MVEPredictor takes no ``task_head_dims`` — the base kwargs hook
        must forward it for the MLP path only, otherwise MVE construction
        would raise ``TypeError``."""
        model = _ToyClassicModel(
            pred_task_head_dims=[8, 4],
            num_endpoints=2,
            uncertainty="mve",
            loss_fn="beta-nll",
        )
        assert isinstance(model.predictor, MVEPredictor)


class TestPredictStepSlicing:
    def test_slices_means_when_uncertainty_is_mve(self):
        model = _ToyClassicModel(num_endpoints=3, uncertainty="mve", loss_fn="beta-nll")
        model.eval()
        batch = _make_batch(batch=4, input_dim=8, num_endpoints=3)
        out = model.predict_step(batch)
        assert out.shape == (4, 3)

    def test_returns_full_output_when_not_mve(self):
        model = _ToyClassicModel(num_endpoints=2)
        model.eval()
        batch = _make_batch(batch=4, input_dim=8, num_endpoints=2)
        out = model.predict_step(batch)
        assert out.shape == (4, 2)


class TestPredictVarianceStep:
    def test_returns_matched_mean_and_log_var_shapes(self):
        model = _ToyClassicModel(num_endpoints=3, uncertainty="mve", loss_fn="beta-nll")
        model.eval()
        batch = _make_batch(batch=5, input_dim=8, num_endpoints=3)
        mean, log_var = model.predict_variance_step(batch)
        assert mean.shape == (5, 3)
        assert log_var.shape == (5, 3)

    def test_mean_matches_predict_step_output(self):
        """``predict_step`` slices the same means that ``predict_variance_step``
        returns; the two must agree under deterministic (eval) forward."""
        model = _ToyClassicModel(num_endpoints=2, uncertainty="mve", loss_fn="beta-nll")
        model.eval()
        batch = _make_batch(batch=3, input_dim=8, num_endpoints=2)
        mean_from_variance, _ = model.predict_variance_step(batch)
        mean_from_predict = model.predict_step(batch)
        assert torch.allclose(mean_from_variance, mean_from_predict)

    def test_raises_when_uncertainty_is_not_mve(self):
        model = _ToyClassicModel(num_endpoints=2)
        batch = _make_batch(batch=3, input_dim=8, num_endpoints=2)
        with pytest.raises(RuntimeError, match="uncertainty='mve'"):
            model.predict_variance_step(batch)


class TestParseLossFnMVEBypass:
    @pytest.mark.parametrize("alias", ["beta-nll", "mve"])
    def test_beta_nll_bypasses_multitask_wrapper_multitask(self, alias):
        model = _ToyClassicModel(num_endpoints=3, uncertainty="mve", loss_fn=alias)
        assert isinstance(model.loss_fn, BetaNLLLoss)
        assert not isinstance(model.loss_fn, MultitaskLoss)

    def test_bounded_beta_nll_bypasses_multitask_wrapper_multitask(self):
        model = _ToyClassicModel(
            num_endpoints=3, uncertainty="mve", loss_fn="bounded-beta-nll"
        )
        assert isinstance(model.loss_fn, BoundedBetaNLLLoss)
        assert not isinstance(model.loss_fn, MultitaskLoss)

    def test_regular_multitask_loss_still_wraps(self):
        """Regression guard: non-MVE multitask paths continue to be wrapped."""
        model = _ToyClassicModel(num_endpoints=3, loss_fn="mse")
        assert isinstance(model.loss_fn, MultitaskLoss)


class TestValidationStepMVESlicing:
    def test_step_completes_without_shape_error_when_uncertainty_is_mve(
        self, monkeypatch
    ):
        """Without the MVE slice, the per-task metric loop would index into
        log-variance columns and mis-associate them with task labels. Runs the
        mixin ``validation_step`` end-to-end with a stubbed ``log`` (no
        Trainer attached) and asserts the loop completes with a finite loss."""
        from matcha.torch.models.mixin import ModelMixin

        model = _ToyClassicModel(num_endpoints=2, uncertainty="mve", loss_fn="beta-nll")
        model.eval()
        monkeypatch.setattr(model, "log", lambda *a, **kw: None)
        batch = _make_batch(batch=6, input_dim=8, num_endpoints=2)
        result = ModelMixin.validation_step(model, batch, 0)
        assert "val_loss" in result
        assert torch.isfinite(result["val_loss"])

    def test_step_bypasses_slice_when_not_mve(self, monkeypatch):
        """Regression guard: the non-MVE path leaves ``y_pred`` untouched."""
        from matcha.torch.models.mixin import ModelMixin

        model = _ToyClassicModel(num_endpoints=2, loss_fn="mse")
        model.eval()
        monkeypatch.setattr(model, "log", lambda *a, **kw: None)
        batch = _make_batch(batch=6, input_dim=8, num_endpoints=2)
        result = ModelMixin.validation_step(model, batch, 0)
        assert "val_loss" in result
        assert torch.isfinite(result["val_loss"])
