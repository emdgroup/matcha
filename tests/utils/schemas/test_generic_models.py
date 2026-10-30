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

from matcha.utils.schemas.generic_models import (
    ChempropFinetunerMixin,
    ClassicMatchaModel,
    FinetunerMixin,
)


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


_FINETUNER_ARGS = dict(
    architecture="ginmodel",
    path_to_pretrained="model",
    pred_hidden_dims=[32],
    activation="relu",
    dropout=0.0,
    num_endpoints=1,
    loss_args={},
    optimizer="adam",
    optimizer_args={"lr": 1e-3},
    pretrain_lr=1e-4,
    pretrain_decay=0.5,
    scheduler="cosine_annealing",
    scheduler_args={"min_lr": 1e-6, "total_steps": 10},
)

_CHEMPROP_FINETUNER_ARGS = dict(
    path_to_pretrained="model",
    num_endpoints=1,
    optimizer="chemprop",
    optimizer_args={"lr": 1e-3},
    scheduler_args={"warmup_epochs": 1, "max_lr": 1e-3, "final_lr": 1e-5},
    pred_hidden_dim=32,
    pred_num_layers=1,
    pred_dropout=0.0,
    pred_activation="relu",
)


class TestFinetunerMVEPairing:
    def test_matcha_finetuner_accepts_beta_nll(self):
        model = FinetunerMixin(
            **_FINETUNER_ARGS,
            uncertainty="mve",
            loss_fn="beta-nll",
        )
        assert model.uncertainty == "mve"

    def test_matcha_finetuner_rejects_non_mve_loss(self):
        with pytest.raises(ValidationError, match="requires loss_fn"):
            FinetunerMixin(
                **_FINETUNER_ARGS,
                uncertainty="mve",
                loss_fn="mse",
            )

    def test_chemprop_finetuner_accepts_only_mve_alias(self):
        model = ChempropFinetunerMixin(
            **_CHEMPROP_FINETUNER_ARGS,
            uncertainty="mve",
            loss_fn="mve",
        )
        assert model.uncertainty == "mve"

        with pytest.raises(ValidationError, match="requires loss_fn"):
            ChempropFinetunerMixin(
                **_CHEMPROP_FINETUNER_ARGS,
                uncertainty="mve",
                loss_fn="beta-nll",
            )


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
