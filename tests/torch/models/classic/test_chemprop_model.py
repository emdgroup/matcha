"""Unit tests for :class:`ChempropModel`'s MVE plumbing.

Stage 2 of issue #99 wires chemprop's ``MveFFN`` + ``MVELoss`` into
:class:`ChempropModel`. The predictor head produces ``(N, T, 2)`` directly,
so :meth:`predict_step` must slice the mean out (``[..., 0]``) and
:meth:`predict_variance_step` must convert the softplus-parameterised
variance to a log-variance before the sklearn ``UncertaintyManager`` reads it.
"""

from collections import namedtuple

import pytest
import torch
from chemprop.nn.predictors import MveFFN, RegressionFFN

from matcha.torch.models.classic.chemprop_model import ChempropModel


# The chemprop batch is a namedtuple-ish; only the first three positions
# (bmg, V_d, X_d) are consumed by MPNN.forward via ``bmg, V_d, X_d, *_ = batch``.
_FakeBatch = namedtuple("_FakeBatch", ["bmg", "V_d", "X_d"])


def _make_chemprop_model(uncertainty: str, num_endpoints: int = 1) -> ChempropModel:
    """Instantiate a tiny :class:`ChempropModel` on CPU."""
    loss_fn = "mve" if uncertainty == "mve" else "mse"
    return ChempropModel(
        enc_atom_hidden_dim=32,
        enc_num_layers=1,
        enc_dropout=0.0,
        pred_hidden_dim=32,
        pred_num_layers=1,
        pred_dropout=0.0,
        num_endpoints=num_endpoints,
        loss_fn=loss_fn,
        uncertainty=uncertainty,
    )


def _stub_ffn_output(model: ChempropModel, N: int, T: int, mve: bool) -> torch.Tensor:
    """Craft the tensor chemprop's FFN would output and stub the model's
    ``forward`` so we can exercise ``predict_step`` / ``predict_variance_step``
    without a real graph batch."""
    if mve:
        # Chemprop's MveFFN emits (N, T, 2): [..., 0] = mean, [..., 1] = softplus(var).
        mean = torch.randn(N, T)
        var = torch.nn.functional.softplus(torch.randn(N, T))
        return torch.stack((mean, var), dim=-1)
    return torch.randn(N, T)


class TestChempropUncertaintyMethod:
    """The ``uncertainty_method`` property is the same source of truth as
    :class:`BaseClassicModel`, so the sklearn ``UncertaintyManager`` can
    dispatch on it identically across model families."""

    def test_default_construction_reports_mc_dropout(self):
        model = _make_chemprop_model(uncertainty="mc-dropout")
        assert model.uncertainty_method == "mc-dropout"

    def test_mve_construction_reports_mve(self):
        model = _make_chemprop_model(uncertainty="mve")
        assert model.uncertainty_method == "mve"


class TestChempropPredictorSelection:
    """The predictor is a chemprop ``MveFFN`` iff ``uncertainty="mve"``,
    a plain ``RegressionFFN`` otherwise. The classifier path uses
    ``BinaryClassificationFFN`` (not covered here: ChempropClassifier is
    untouched by issue #99)."""

    def test_mve_selects_mve_ffn(self):
        model = _make_chemprop_model(uncertainty="mve")
        assert isinstance(model.predictor, MveFFN)

    def test_default_selects_regression_ffn(self):
        model = _make_chemprop_model(uncertainty="mc-dropout")
        assert isinstance(model.predictor, RegressionFFN)


class TestChempropPredictStepMVE:
    """When configured for MVE, ``predict_step`` must return only the mean
    (``[..., 0]`` of the ``(N, T, 2)`` tensor emitted by ``MveFFN``)."""

    def test_predict_step_returns_mean_only(self, monkeypatch):
        N, T = 5, 3
        model = _make_chemprop_model(uncertainty="mve", num_endpoints=T)
        model.eval()
        stub = _stub_ffn_output(model, N, T, mve=True)
        # Bypass the real message-passing forward — chemprop's MPNN.predict_step
        # is ``bmg, V_d, X_d, *_ = batch; return self(bmg, V_d, X_d)``.
        monkeypatch.setattr(model, "forward", lambda *a, **kw: stub)
        batch = _FakeBatch(bmg=None, V_d=None, X_d=None)
        out = model.predict_step(batch, batch_idx=0)
        assert out.shape == (N, T)
        assert torch.allclose(out, stub[..., 0])

    def test_predict_step_no_mve_returns_full_output(self, monkeypatch):
        N, T = 5, 3
        model = _make_chemprop_model(uncertainty="mc-dropout", num_endpoints=T)
        model.eval()
        stub = _stub_ffn_output(model, N, T, mve=False)
        monkeypatch.setattr(model, "forward", lambda *a, **kw: stub)
        batch = _FakeBatch(bmg=None, V_d=None, X_d=None)
        out = model.predict_step(batch, batch_idx=0)
        assert out.shape == (N, T)
        assert torch.allclose(out, stub)


class TestChempropPredictVarianceStep:
    """``predict_variance_step`` converts chemprop's softplus-variance into a
    log-variance, matching matcha's convention so
    :meth:`UncertaintyManager._compute_mve` can exponentiate it back to a
    variance and unscale it through the target scaler."""

    def test_returns_mean_and_log_var_shapes(self, monkeypatch):
        N, T = 5, 3
        model = _make_chemprop_model(uncertainty="mve", num_endpoints=T)
        model.eval()
        stub = _stub_ffn_output(model, N, T, mve=True)
        monkeypatch.setattr(model, "forward", lambda *a, **kw: stub)
        batch = _FakeBatch(bmg=None, V_d=None, X_d=None)
        mean, log_var = model.predict_variance_step(batch)
        assert mean.shape == (N, T)
        assert log_var.shape == (N, T)

    def test_log_var_finite_after_softplus_to_log_conversion(self, monkeypatch):
        N, T = 5, 3
        model = _make_chemprop_model(uncertainty="mve", num_endpoints=T)
        model.eval()
        stub = _stub_ffn_output(model, N, T, mve=True)
        monkeypatch.setattr(model, "forward", lambda *a, **kw: stub)
        batch = _FakeBatch(bmg=None, V_d=None, X_d=None)
        _, log_var = model.predict_variance_step(batch)
        assert torch.all(torch.isfinite(log_var))

    def test_log_var_matches_manual_conversion(self, monkeypatch):
        N, T = 4, 2
        model = _make_chemprop_model(uncertainty="mve", num_endpoints=T)
        model.eval()
        stub = _stub_ffn_output(model, N, T, mve=True)
        monkeypatch.setattr(model, "forward", lambda *a, **kw: stub)
        batch = _FakeBatch(bmg=None, V_d=None, X_d=None)
        mean, log_var = model.predict_variance_step(batch)
        expected_mean = stub[..., 0]
        expected_log_var = torch.log(stub[..., 1].clamp_min(1e-6))
        assert torch.allclose(mean, expected_mean)
        assert torch.allclose(log_var, expected_log_var)

    def test_raises_when_not_mve(self):
        model = _make_chemprop_model(uncertainty="mc-dropout")
        batch = _FakeBatch(bmg=None, V_d=None, X_d=None)
        with pytest.raises(RuntimeError, match="uncertainty='mve'"):
            model.predict_variance_step(batch)
