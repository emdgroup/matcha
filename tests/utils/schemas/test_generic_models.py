"""Tests for the ``ClassicMatchaModel`` uncertainty enum and MVE-pairing validator.

The schema on :class:`matcha.utils.schemas.generic_models.ClassicMatchaModel`
enforces two orthogonal concerns:

* ``uncertainty`` is an explicit enum (``Literal["mc-dropout", "mve"]``) that
  defaults to ``"mc-dropout"``. Legacy configs with ``uncertainty=None`` are
  normalized to ``"mc-dropout"`` for backwards compatibility.
* Three symmetric MVE / loss pairing rules:

  1. ``uncertainty="mve"`` requires a MVE-family ``loss_fn``.
  2. A MVE-family ``loss_fn`` requires ``uncertainty="mve"``.
  3. ``uncertainty="mve"`` is incompatible with ``multitask`` / ``multiloss`` /
     ``gradnorm``.
"""

import pytest
from pydantic import ValidationError

from matcha.utils.schemas.generic_models import ClassicMatchaModel


_BASE_ARGS = dict(
    additional_mol_features_dim=0,
    num_endpoints=1,
    loss_args={},
    optimizer="adam",
    optimizer_args={"lr": 1e-3},
    scheduler="cosine_annealing",
    scheduler_args={"min_lr": 1e-6, "total_steps": 50},
)


class TestClassicMatchaModelUncertaintyEnum:
    """Explicit enum surface and backwards compatibility with legacy ``None``."""

    def test_default_uncertainty_is_mc_dropout(self):
        m = ClassicMatchaModel(loss_fn="mse", **_BASE_ARGS)
        assert m.uncertainty == "mc-dropout"

    def test_explicit_mc_dropout_accepted(self):
        m = ClassicMatchaModel(loss_fn="mse", uncertainty="mc-dropout", **_BASE_ARGS)
        assert m.uncertainty == "mc-dropout"

    def test_legacy_none_normalized_to_mc_dropout(self):
        m = ClassicMatchaModel(loss_fn="mse", uncertainty=None, **_BASE_ARGS)
        assert m.uncertainty == "mc-dropout"

    @pytest.mark.parametrize("bad", ["foo", "dropout", "MVE", "mc_dropout"])
    def test_invalid_literal_rejected(self, bad):
        with pytest.raises(ValidationError):
            ClassicMatchaModel(loss_fn="mse", uncertainty=bad, **_BASE_ARGS)


class TestClassicMatchaModelMVEPairing:
    """Symmetric validation rules on the MVE / loss pairing."""

    # -- happy paths -----------------------------------------------------

    @pytest.mark.parametrize("loss", ["beta-nll", "mve", "bounded-beta-nll"])
    def test_mve_uncertainty_with_matching_loss(self, loss):
        m = ClassicMatchaModel(loss_fn=loss, uncertainty="mve", **_BASE_ARGS)
        assert m.uncertainty == "mve"
        assert m.loss_fn == loss

    # -- rule 1: uncertainty="mve" requires MVE loss ---------------------

    @pytest.mark.parametrize("loss", ["mse", "mae", "huber", "bce"])
    def test_mve_uncertainty_rejects_non_mve_loss(self, loss):
        with pytest.raises(ValidationError, match=r"requires loss_fn in"):
            ClassicMatchaModel(loss_fn=loss, uncertainty="mve", **_BASE_ARGS)

    # -- rule 2: MVE loss requires uncertainty="mve" ---------------------

    @pytest.mark.parametrize("loss", ["beta-nll", "mve", "bounded-beta-nll"])
    def test_mve_loss_without_mve_uncertainty_rejected(self, loss):
        with pytest.raises(ValidationError, match=r"requires uncertainty='mve'"):
            ClassicMatchaModel(loss_fn=loss, **_BASE_ARGS)

    # -- rule 3: MVE + multitask / multiloss / gradnorm incompatible -----

    @pytest.mark.parametrize("loss", ["gradnorm", "multiloss", "multitask"])
    def test_mve_uncertainty_rejects_multi_task_loss(self, loss):
        with pytest.raises(ValidationError, match=r"requires loss_fn in"):
            ClassicMatchaModel(loss_fn=loss, uncertainty="mve", **_BASE_ARGS)
