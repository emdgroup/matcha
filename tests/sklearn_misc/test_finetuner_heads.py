"""Finetuner objective-transition and encoder-only head tests."""

import os
import warnings

import numpy as np
import pytest
import torch

from matcha.sklearn.finetuner import FinetuningClassifier, FinetuningRegressor

from .conftest import (
    FINETUNE_TRAIN,
    FINETUNE_TRAIN_NO_OVERRIDE,
    make_chemprop_classifier,
    make_chemprop_regressor,
)


# =========================================================================
# FFN type mismatch fix (issue #400, stage 2)
# =========================================================================


class TestFFNTypeMismatchFix:
    """Verify FFN type replacement when pred_hidden_dims=None and objective changes.

    When a pretrained regression model is finetuned on a classification objective
    (or vice versa) with pred_hidden_dims=None (Case C: keep pretrained FFN, resize
    output), the predictor FFN must be replaced with the correct type while
    transferring compatible weights.
    """

    @pytest.fixture()
    def chemprop_pretrained_regression_path(self, mol_list, regression_y, tmp_path):
        """Fit a base ChempropRegressor, save it, return its path."""
        model = make_chemprop_regressor()
        save_dir = str(tmp_path / "pretrained_reg")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture()
    def chemprop_pretrained_classifier_path(self, mol_list, classification_y, tmp_path):
        """Fit a base ChempropClassifier, save it, return its path."""
        model = make_chemprop_classifier()
        save_dir = str(tmp_path / "pretrained_cls")
        model.fit(mol_list, classification_y)
        model.save_model(save_dir)
        return save_dir

    def test_regression_to_classification_no_hidden_dims(
        self, chemprop_pretrained_regression_path, mol_list, classification_y
    ):
        """Pretrain regression → finetune classification with pred_hidden_dims=None.

        Case C with FFN type mismatch: the predictor should be replaced with
        BinaryClassificationFFN and predict_proba() outputs should be valid
        probabilities in [0, 1].
        """
        from chemprop.nn.predictors import BinaryClassificationFFN

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finetuner = FinetuningClassifier(
                path_to_pretrained=chemprop_pretrained_regression_path,
                **FINETUNE_TRAIN_NO_OVERRIDE,
            )

        # Should have emitted a warning about FFN type replacement
        ffn_warnings = [w for w in caught if "Replacing predictor" in str(w.message)]
        assert len(ffn_warnings) == 1

        finetuner.fit(mol_list, classification_y)
        proba = finetuner.predict_proba(mol_list)

        assert proba.shape == (len(mol_list), 1)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

        # Verify the underlying predictor is the correct type
        assert isinstance(finetuner._model.predictor, BinaryClassificationFFN)

    def test_same_objective_no_change(
        self, chemprop_pretrained_regression_path, mol_list, regression_y
    ):
        """Pretrain regression → finetune regression with pred_hidden_dims=None.

        Case C without type mismatch: the predictor class should remain
        RegressionFFN (no unnecessary replacement).
        """
        from chemprop.nn.predictors import RegressionFFN

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finetuner = FinetuningRegressor(
                path_to_pretrained=chemprop_pretrained_regression_path,
                **FINETUNE_TRAIN_NO_OVERRIDE,
            )

        # No FFN type warning should be emitted
        ffn_warnings = [w for w in caught if "Replacing predictor" in str(w.message)]
        assert len(ffn_warnings) == 0

        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)

        assert preds.shape == (len(mol_list), 1)

        # Verify predictor is still RegressionFFN
        assert isinstance(finetuner._model.predictor, RegressionFFN)

    def test_regression_to_classification_mismatched_endpoints(
        self, mol_list, multitask_regression_y, classification_y, tmp_path
    ):
        """Pretrain multi-endpoint regression → finetune single-endpoint classification.

        This is the exact failure case from issue #405: the pretrained model has
        n_tasks=N (e.g., 3) and the finetuned model has n_tasks=1, causing a
        size mismatch in the final output layer when both old and new state dicts
        contain the same key but with incompatible shapes.
        """
        from chemprop.nn.predictors import BinaryClassificationFFN

        # Pretrain a multi-endpoint regression model (n_tasks=2)
        from matcha.sklearn import ChempropRegressor

        pretrained = ChempropRegressor(
            enc_num_layers=1,
            enc_atom_hidden_dim=32,
            pred_hidden_dim=32,
            pred_num_layers=1,
            feature_list=None,
            num_endpoints=2,
            num_epochs=1,
            batch_size=32,
            accelerator="cpu",
            devices=1,
            early_stopping=False,
            stochastic_weight_averaging=False,
        )
        pretrained.fit(mol_list, multitask_regression_y)
        save_dir = str(tmp_path / "pretrained_multitask_reg")
        pretrained.save_model(save_dir)

        # Finetune as single-endpoint binary classification (n_tasks=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finetuner = FinetuningClassifier(
                path_to_pretrained=save_dir,
                **FINETUNE_TRAIN_NO_OVERRIDE,
            )

        # Should have emitted a warning about FFN type replacement
        ffn_warnings = [w for w in caught if "Replacing predictor" in str(w.message)]
        assert len(ffn_warnings) == 1

        finetuner.fit(mol_list, classification_y)
        proba = finetuner.predict_proba(mol_list)

        # Verify correct output shape and valid probabilities
        assert proba.shape == (len(mol_list), 1)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

        # Verify the underlying predictor is the correct type
        assert isinstance(finetuner._model.predictor, BinaryClassificationFFN)

        # Verify hidden layer weights were transferred (not randomly initialized)
        # The first hidden layer should have non-zero weights from pretraining
        first_layer = finetuner._model.predictor.ffn[0][0]
        assert first_layer.weight.abs().sum() > 0

    def test_classification_to_regression_no_hidden_dims(
        self, chemprop_pretrained_classifier_path, mol_list, regression_y
    ):
        """Pretrain classification → finetune regression with pred_hidden_dims=None.

        Case C with FFN type mismatch: the predictor should be replaced with
        RegressionFFN and outputs should be unbounded (no sigmoid activation).
        """
        from chemprop.nn.predictors import RegressionFFN

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finetuner = FinetuningRegressor(
                path_to_pretrained=chemprop_pretrained_classifier_path,
                **FINETUNE_TRAIN_NO_OVERRIDE,
            )

        # Should have emitted a warning about FFN type replacement
        ffn_warnings = [w for w in caught if "Replacing predictor" in str(w.message)]
        assert len(ffn_warnings) == 1

        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)

        assert preds.shape == (len(mol_list), 1)

        # Verify the underlying predictor is RegressionFFN
        assert isinstance(finetuner._model.predictor, RegressionFFN)


# =========================================================================
# Linear head for encoder-only checkpoints (issue #402, stage 3)
# =========================================================================


class TestLinearHead:
    """Verify linear head creation for encoder-only chemprop checkpoints.

    When finetuning from an encoder-only checkpoint (BondMessagePassing weights
    only), pred_hidden_dims=None should produce a linear head (Case A), and
    pred_hidden_dims=[int] should produce a custom FFN (Case B).
    """

    @pytest.fixture()
    def encoder_only_checkpoint_path(self, mol_list, regression_y, tmp_path):
        """Create a pretrained directory with an encoder-only BMP checkpoint.

        Saves a full ChempropRegressor to produce valid config/state yamls,
        then replaces model.ckpt with an encoder-only format containing
        'hyper_parameters' and 'state_dict' keys (raw BondMessagePassing
        weights). The test methods use monkeypatch to force the fallback
        loading path which sets self.predictor = None.
        """
        from chemprop.nn import BondMessagePassing

        model = make_chemprop_regressor()
        save_dir = str(tmp_path / "encoder_only_chemprop")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)

        # Replace with encoder-only checkpoint format
        mp = BondMessagePassing(d_h=32, depth=1)
        ckpt = {
            "hyper_parameters": {"d_h": 32, "depth": 1},
            "state_dict": mp.state_dict(),
        }
        torch.save(ckpt, os.path.join(save_dir, "model.ckpt"))
        return save_dir

    def _make_finetuner(self, path, finetuner_cls, train_kwargs, monkeypatch):
        """Create a finetuner, forcing the encoder-only fallback loading path.

        Patches torch.load so the first call (weights_only=False) raises,
        simulating an incompatible checkpoint format. The fallback call
        (weights_only=True) proceeds normally.
        """
        original_torch_load = torch.load
        call_count = [0]

        def patched_load(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: simulate incompatible format
                raise RuntimeError("Simulated incompatible checkpoint format")
            return original_torch_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", patched_load)
        finetuner = finetuner_cls(path_to_pretrained=path, **train_kwargs)
        monkeypatch.undo()
        return finetuner

    def test_linear_head_regression(
        self, encoder_only_checkpoint_path, mol_list, regression_y, monkeypatch
    ):
        """Encoder-only + pred_hidden_dims=None → linear head (Case A, regression)."""
        from chemprop.nn.predictors import RegressionFFN

        finetuner = self._make_finetuner(
            encoder_only_checkpoint_path,
            FinetuningRegressor,
            FINETUNE_TRAIN_NO_OVERRIDE,
            monkeypatch,
        )

        # Verify predictor is a RegressionFFN with a single linear layer
        predictor = finetuner._model.predictor
        assert isinstance(predictor, RegressionFFN)
        # n_layers=0 produces a single-block FFN: [Linear(input_dim, output_dim)]
        assert len(predictor.ffn) == 1

        # Verify fit + predict works
        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(preds))

    def test_linear_head_mve(self, encoder_only_checkpoint_path, monkeypatch):
        """Encoder-only + no hidden dims builds a linear MVE head."""
        from chemprop.nn.predictors import MveFFN

        train_kwargs = {
            **FINETUNE_TRAIN_NO_OVERRIDE,
            "uncertainty": "mve",
            "loss_fn": "mve",
        }
        finetuner = self._make_finetuner(
            encoder_only_checkpoint_path,
            FinetuningRegressor,
            train_kwargs,
            monkeypatch,
        )

        predictor = finetuner._model.predictor
        output = predictor(torch.randn(4, predictor.ffn.input_dim))
        assert isinstance(predictor, MveFFN)
        assert len(predictor.ffn) == 1
        assert output.shape == (4, 1, 2)

    def test_linear_head_classification(
        self, encoder_only_checkpoint_path, mol_list, classification_y, monkeypatch
    ):
        """Encoder-only + pred_hidden_dims=None → linear head (Case A, classification)."""
        from chemprop.nn.predictors import BinaryClassificationFFN

        finetuner = self._make_finetuner(
            encoder_only_checkpoint_path,
            FinetuningClassifier,
            FINETUNE_TRAIN_NO_OVERRIDE,
            monkeypatch,
        )

        # Verify predictor is a BinaryClassificationFFN with a single linear layer
        predictor = finetuner._model.predictor
        assert isinstance(predictor, BinaryClassificationFFN)
        assert len(predictor.ffn) == 1

        # Verify fit + predict works
        finetuner.fit(mol_list, classification_y)
        proba = finetuner.predict_proba(mol_list)
        assert proba.shape == (len(mol_list), 1)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_custom_mve_ffn_from_encoder_only(
        self, encoder_only_checkpoint_path, monkeypatch
    ):
        """Encoder-only + hidden dims builds a custom MVE head."""
        from chemprop.nn.predictors import MveFFN

        train_kwargs = {
            **FINETUNE_TRAIN,
            "uncertainty": "mve",
            "loss_fn": "mve",
        }
        finetuner = self._make_finetuner(
            encoder_only_checkpoint_path,
            FinetuningRegressor,
            train_kwargs,
            monkeypatch,
        )

        predictor = finetuner._model.predictor
        output = predictor(torch.randn(4, predictor.ffn.input_dim))
        assert isinstance(predictor, MveFFN)
        assert len(predictor.ffn) > 1
        assert output.shape == (4, 1, 2)

    def test_custom_ffn_from_encoder_only(
        self, encoder_only_checkpoint_path, mol_list, regression_y, monkeypatch
    ):
        """Encoder-only + pred_hidden_dims=[int] → custom FFN (Case B)."""
        from chemprop.nn.predictors import RegressionFFN

        train_kwargs = {
            k: v for k, v in FINETUNE_TRAIN.items() if k != "pred_hidden_dims"
        }
        train_kwargs["pred_hidden_dims"] = [64]

        finetuner = self._make_finetuner(
            encoder_only_checkpoint_path,
            FinetuningRegressor,
            train_kwargs,
            monkeypatch,
        )

        # Verify predictor is a RegressionFFN with multiple layers
        predictor = finetuner._model.predictor
        assert isinstance(predictor, RegressionFFN)
        # Should have more than 1 block (input → hidden → output)
        assert len(predictor.ffn) > 1

        # Verify fit + predict works
        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(preds))
