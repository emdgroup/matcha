"""Test FinetuningRegressor and FinetuningClassifier for all modalities.

Exercises: construct a finetuner from a saved pretrained model, fit, and predict.
Each modality (tabular, graph, CLM, chemprop) is tested with both
a regressor and a classifier.

The finetuner workflow is:
1. Create and fit a base model (the "pretrained" model).
2. Save it to a folder.
3. Construct a FinetuningRegressor / FinetuningClassifier pointing at that folder.
4. Fit the finetuner on the same data.
5. Predict and verify outputs.

Runtime optimisation: pretrained-model saving is performed once per factory
via fixtures.  The finetuner fit is also performed once per factory; multiple
assertions reuse the same fitted finetuner.
"""

import os
import shutil
import warnings

import numpy as np
import pytest
import torch

from matcha.sklearn.finetuner import FinetuningClassifier, FinetuningRegressor

from .conftest import (
    make_mlp_regressor,
    make_mlp_classifier,
    make_gin_regressor,
    make_gin_classifier,
    make_cnn_regressor,
    make_cnn_classifier,
    make_chemprop_regressor,
    make_chemprop_classifier,
)


# =========================================================================
# Shared finetune training kwargs
# =========================================================================

FINETUNE_TRAIN = dict(
    num_epochs=1,
    batch_size=32,
    accelerator="cpu",
    devices=1,
    early_stopping=False,
    stochastic_weight_averaging=False,
    pred_hidden_dims=[32],
    finetuning_strategy="full",
)


# =========================================================================
# Parametrized factories (local, reusing conftest helpers)
# =========================================================================

_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLPRegressor"),
    pytest.param(make_gin_regressor, id="GINRegressor"),
    pytest.param(make_cnn_regressor, id="CNNRegressor"),
    pytest.param(make_chemprop_regressor, id="ChempropRegressor"),
]

_CLASSIFIER_FACTORIES = [
    pytest.param(make_mlp_classifier, id="MLPClassifier"),
    pytest.param(make_gin_classifier, id="GINClassifier"),
    pytest.param(make_cnn_classifier, id="CNNClassifier"),
    pytest.param(make_chemprop_classifier, id="ChempropClassifier"),
]


# =========================================================================
# Fixtures – pretrained paths  (fit base model once, save, yield path)
# =========================================================================


@pytest.fixture(params=_REGRESSOR_FACTORIES)
def pretrained_regressor_path(request, mol_list, regression_y, tmp_path):
    """Fit a base regressor, save it, and return the save path."""
    factory = request.param
    model = factory()
    save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
    model.fit(mol_list, regression_y)
    model.save_model(save_dir)
    return save_dir


@pytest.fixture(params=_CLASSIFIER_FACTORIES)
def pretrained_classifier_path(request, mol_list, classification_y, tmp_path):
    """Fit a base classifier, save it, and return the save path."""
    factory = request.param
    model = factory()
    save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
    model.fit(mol_list, classification_y)
    model.save_model(save_dir)
    return save_dir


# =========================================================================
# Fixtures – fitted finetuners  (one fit per factory)
# =========================================================================


@pytest.fixture()
def fitted_finetuning_regressor(pretrained_regressor_path, mol_list, regression_y):
    """Construct and fit a FinetuningRegressor; reuse for predict tests."""
    finetuner = FinetuningRegressor(
        path_to_pretrained=pretrained_regressor_path,
        **FINETUNE_TRAIN,
    )
    finetuner.fit(mol_list, regression_y)
    return finetuner


@pytest.fixture()
def fitted_finetuning_classifier(
    pretrained_classifier_path, mol_list, classification_y
):
    """Construct and fit a FinetuningClassifier; reuse for predict tests."""
    finetuner = FinetuningClassifier(
        path_to_pretrained=pretrained_classifier_path,
        **FINETUNE_TRAIN,
    )
    finetuner.fit(mol_list, classification_y)
    return finetuner


# =========================================================================
# FinetuningRegressor – construction  (no fit needed)
# =========================================================================


class TestFinetuningRegressorConstruction:
    """Verify that a FinetuningRegressor can be constructed from a pretrained path."""

    def test_construction_succeeds(self, pretrained_regressor_path):
        finetuner = FinetuningRegressor(
            path_to_pretrained=pretrained_regressor_path,
            **FINETUNE_TRAIN,
        )
        assert finetuner is not None

    def test_model_is_not_none(self, pretrained_regressor_path):
        finetuner = FinetuningRegressor(
            path_to_pretrained=pretrained_regressor_path,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model is not None


# =========================================================================
# FinetuningRegressor – fit + predict  (one fit via fixture)
# =========================================================================


class TestFinetuningRegressorPredict:
    """Verify FinetuningRegressor predict returns well-formed output."""

    def test_is_fitted_after_fit(self, fitted_finetuning_regressor):
        assert fitted_finetuning_regressor.is_fitted is True

    def test_predict_returns_ndarray(self, fitted_finetuning_regressor, mol_list):
        preds = fitted_finetuning_regressor.predict(mol_list)
        assert isinstance(preds, np.ndarray)

    def test_predict_shape_matches_input(self, fitted_finetuning_regressor, mol_list):
        preds = fitted_finetuning_regressor.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_predict_values_are_finite(self, fitted_finetuning_regressor, mol_list):
        preds = fitted_finetuning_regressor.predict(mol_list)
        assert np.all(np.isfinite(preds))


# =========================================================================
# FinetuningClassifier – construction  (no fit needed)
# =========================================================================


class TestFinetuningClassifierConstruction:
    """Verify that a FinetuningClassifier can be constructed from a pretrained path."""

    def test_construction_succeeds(self, pretrained_classifier_path):
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_classifier_path,
            **FINETUNE_TRAIN,
        )
        assert finetuner is not None

    def test_model_is_not_none(self, pretrained_classifier_path):
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_classifier_path,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model is not None


# =========================================================================
# FinetuningClassifier – fit + predict  (one fit via fixture)
# =========================================================================


class TestFinetuningClassifierPredict:
    """Verify FinetuningClassifier predict returns well-formed output."""

    def test_is_fitted_after_fit(self, fitted_finetuning_classifier):
        assert fitted_finetuning_classifier.is_fitted is True

    def test_predict_returns_ndarray(self, fitted_finetuning_classifier, mol_list):
        preds = fitted_finetuning_classifier.predict(mol_list)
        assert isinstance(preds, np.ndarray)

    def test_predict_shape_matches_input(self, fitted_finetuning_classifier, mol_list):
        preds = fitted_finetuning_classifier.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_predict_values_are_binary(self, fitted_finetuning_classifier, mol_list):
        preds = fitted_finetuning_classifier.predict(mol_list)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0.0, 1.0})

    def test_predict_proba_shape(self, fitted_finetuning_classifier, mol_list):
        proba = fitted_finetuning_classifier.predict_proba(mol_list)
        assert proba.shape == (len(mol_list), 1)

    def test_predict_proba_in_0_1_range(self, fitted_finetuning_classifier, mol_list):
        proba = fitted_finetuning_classifier.predict_proba(mol_list)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


# =========================================================================
# Finetuner scheduler behavior — pretrain params have no scheduler
# =========================================================================


_NON_CHEMPROP_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLPRegressor"),
    pytest.param(make_gin_regressor, id="GINRegressor"),
    pytest.param(make_cnn_regressor, id="CNNRegressor"),
]


@pytest.fixture(params=_NON_CHEMPROP_REGRESSOR_FACTORIES)
def pretrained_non_chemprop_path(request, mol_list, regression_y, tmp_path):
    """Fit a non-Chemprop regressor, save it, return the path."""
    factory = request.param
    model = factory()
    save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
    model.fit(mol_list, regression_y)
    model.save_model(save_dir)
    return save_dir


class TestFinetunerNoPretrainScheduler:
    """Verify that full finetuning does not create a pretrain_scheduler.

    After Stage 2 of issue #361, pretrained param groups keep their constant
    layer-wise decayed LRs with no scheduler driving them toward a shared minimum.
    Chemprop is excluded as it manages its own scheduler internally.
    """

    def test_full_finetuner_has_no_pretrain_scheduler(
        self, pretrained_non_chemprop_path
    ):
        """After construction with strategy='full', pretrain_scheduler should not exist."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=pretrained_non_chemprop_path,
            **FINETUNE_TRAIN,
        )
        assert not hasattr(finetuner._model, "pretrain_scheduler")

    def test_full_finetuner_has_predictor_scheduler(self, pretrained_non_chemprop_path):
        """After construction with strategy='full', predictor_scheduler should exist."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=pretrained_non_chemprop_path,
            **FINETUNE_TRAIN,
        )
        assert hasattr(finetuner._model, "predictor_scheduler")

    def test_pretrain_lr_stays_constant_during_training(
        self, pretrained_non_chemprop_path, mol_list, regression_y
    ):
        """Pretrain param group LRs should remain constant (not scheduled) after fit."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=pretrained_non_chemprop_path,
            **FINETUNE_TRAIN,
        )
        # Record pretrain optimizer LRs before training
        model = finetuner._model
        initial_lrs = [g["lr"] for g in model.pretrain_optimizer.param_groups]

        finetuner.fit(mol_list, regression_y)

        # After training, pretrain LRs should be unchanged (no scheduler)
        final_lrs = [g["lr"] for g in model.pretrain_optimizer.param_groups]
        assert initial_lrs == final_lrs


# =========================================================================
# Self-contained loading integration tests (issue #373, stage 3)
# =========================================================================

_SELF_CONTAINED_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLP"),
    pytest.param(make_gin_regressor, id="GIN"),
]


class TestSelfContainedLoading:
    """Verify finetuned models are self-contained and path-independent.

    After saving, a finetuned model must be loadable without access to
    ancestor model directories.
    """

    @pytest.fixture(params=_SELF_CONTAINED_REGRESSOR_FACTORIES)
    def pretrained_path(self, request, mol_list, regression_y, tmp_path):
        """Fit a base regressor, save it, return its path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture()
    def fitted_and_saved_finetuner(
        self, pretrained_path, mol_list, regression_y, tmp_path
    ):
        """Fit a finetuner, save it, return (save_path, expected_preds)."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=pretrained_path,
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds_before = finetuner.predict(mol_list)

        save_dir = str(tmp_path / "finetuned_saved")
        finetuner.save_model(save_dir)
        return save_dir, preds_before, pretrained_path

    def test_load_after_source_deleted(self, fitted_and_saved_finetuner, mol_list):
        """Load a saved finetuner after deleting the pretrained source path."""
        save_dir, preds_before, pretrained_path = fitted_and_saved_finetuner

        # Delete the pretrained source
        shutil.rmtree(pretrained_path)
        assert not os.path.exists(pretrained_path)

        # Load from saved path — should not need the pretrained source
        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)

    def test_load_from_moved_directory(
        self, fitted_and_saved_finetuner, mol_list, tmp_path
    ):
        """Load a saved finetuner after moving its directory to a new path."""
        save_dir, preds_before, _ = fitted_and_saved_finetuner

        # Move the saved model to a different location
        new_dir = str(tmp_path / "moved_model")
        shutil.move(save_dir, new_dir)
        assert not os.path.exists(save_dir)

        # Load from new path
        loaded = FinetuningRegressor.from_folder(new_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)

    def test_save_load_roundtrip_regression(self, fitted_and_saved_finetuner, mol_list):
        """Regression test: basic save/load round-trip produces same predictions."""
        save_dir, preds_before, _ = fitted_and_saved_finetuner

        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)


_NESTED_FINETUNER_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLP"),
    pytest.param(make_gin_regressor, id="GIN"),
]


class TestNestedSelfContainedLoading:
    """Verify nested finetuning (A → B → C) produces self-contained models.

    After saving model C, removing A and B paths must not break loading.
    """

    @pytest.fixture(params=_NESTED_FINETUNER_FACTORIES)
    def nested_finetuner_artifacts(self, request, mol_list, regression_y, tmp_path):
        """Build a three-level finetuning chain: base → B → C, save C."""
        factory = request.param

        # Level A: base model
        base = factory()
        base_dir = str(tmp_path / "base_A")
        base.fit(mol_list, regression_y)
        base.save_model(base_dir)

        # Level B: finetuned from A
        finetuner_b = FinetuningRegressor(
            path_to_pretrained=base_dir,
            **FINETUNE_TRAIN,
        )
        finetuner_b.fit(mol_list, regression_y)
        b_dir = str(tmp_path / "finetuned_B")
        finetuner_b.save_model(b_dir)

        # Level C: finetuned from B
        finetuner_c = FinetuningRegressor(
            path_to_pretrained=b_dir,
            **FINETUNE_TRAIN,
        )
        finetuner_c.fit(mol_list, regression_y)
        preds_c = finetuner_c.predict(mol_list)
        c_dir = str(tmp_path / "finetuned_C")
        finetuner_c.save_model(c_dir)

        return c_dir, preds_c, base_dir, b_dir

    def test_nested_load_after_ancestors_deleted(
        self, nested_finetuner_artifacts, mol_list
    ):
        """Load model C after deleting both A and B source directories."""
        c_dir, preds_c, base_dir, b_dir = nested_finetuner_artifacts

        # Remove ancestor paths
        shutil.rmtree(base_dir)
        shutil.rmtree(b_dir)
        assert not os.path.exists(base_dir)
        assert not os.path.exists(b_dir)

        # Load C — must work without A or B
        loaded = FinetuningRegressor.from_folder(c_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_loaded, preds_c, rtol=1e-5, atol=1e-6)


# =========================================================================
# Self-contained loading for ChempropFinetuner (issue #396, stage 3)
# =========================================================================


class TestSelfContainedChempropLoading:
    """Verify ChempropFinetuner-backed models are self-contained after save.

    After saving, loading must work without access to the original pretrained
    Chemprop model directory.
    """

    @pytest.fixture()
    def chemprop_pretrained_path(self, mol_list, regression_y, tmp_path):
        """Fit a base ChempropRegressor, save it, return its path."""
        model = make_chemprop_regressor()
        save_dir = str(tmp_path / "pretrained_chemprop")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture()
    def fitted_chemprop_finetuner(
        self, chemprop_pretrained_path, mol_list, regression_y, tmp_path
    ):
        """Fit a ChempropFinetuner, save it, return (save_path, preds, pretrained_path)."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=chemprop_pretrained_path,
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds_before = finetuner.predict(mol_list)

        save_dir = str(tmp_path / "finetuned_chemprop")
        finetuner.save_model(save_dir)
        return save_dir, preds_before, chemprop_pretrained_path

    def test_model_yaml_has_sentinel(self, fitted_chemprop_finetuner):
        """model.yaml should contain path_to_pretrained: __self_contained__."""
        from matcha.utils.serialization import load_yaml

        save_dir, _, _ = fitted_chemprop_finetuner
        model_yaml = load_yaml(os.path.join(save_dir, "config", "model.yaml"))
        assert model_yaml["path_to_pretrained"] == "__self_contained__"

    def test_pretrain_config_yaml_exists(self, fitted_chemprop_finetuner):
        """pretrain_config.yaml should be written to the config directory."""
        from matcha.utils.serialization import load_yaml

        save_dir, _, _ = fitted_chemprop_finetuner
        config_path = os.path.join(save_dir, "config", "pretrain_config.yaml")
        assert os.path.exists(config_path)

        config = load_yaml(config_path)
        assert config["origin_type"] == "chemprop"
        assert "pretrain_params" in config

    def test_load_after_source_deleted(self, fitted_chemprop_finetuner, mol_list):
        """Load a saved ChempropFinetuner after deleting the pretrained source."""
        save_dir, preds_before, pretrained_path = fitted_chemprop_finetuner

        # Delete the pretrained source
        shutil.rmtree(pretrained_path)
        assert not os.path.exists(pretrained_path)

        # Load — should not need the pretrained source
        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)

    def test_load_from_moved_directory(
        self, fitted_chemprop_finetuner, mol_list, tmp_path
    ):
        """Load a saved ChempropFinetuner after moving its directory."""
        save_dir, preds_before, _ = fitted_chemprop_finetuner

        new_dir = str(tmp_path / "moved_chemprop_model")
        shutil.move(save_dir, new_dir)
        assert not os.path.exists(save_dir)

        loaded = FinetuningRegressor.from_folder(new_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)


# =========================================================================
# FFN type mismatch fix (issue #400, stage 2)
# =========================================================================

# pred_hidden_dims=None with a full checkpoint triggers Case C (resize output layer)
FINETUNE_TRAIN_NO_OVERRIDE = {
    k: v for k, v in FINETUNE_TRAIN.items() if k != "pred_hidden_dims"
}
FINETUNE_TRAIN_NO_OVERRIDE["pred_hidden_dims"] = None


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


# =========================================================================
# Label encoder preservation in FinetuningClassifier (issue #407, stage 1)
# =========================================================================

_ALL_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLP"),
    pytest.param(make_gin_regressor, id="GIN"),
    pytest.param(make_cnn_regressor, id="CNN"),
    pytest.param(make_chemprop_regressor, id="Chemprop"),
]


class TestFinetuningClassifierLabelEncoder:
    """Verify label encoder params are preserved through FinetuningClassifier construction.

    When finetuning regression → classification with user-provided class_thresholds
    and class_labels, the label encoder must retain those parameters so that
    predictions produce categorical outputs.
    """

    LABEL_ENCODER_PARAMS = {
        "encoder_type": "binary_classification",
        0: {
            "task_label": "activity",
            "class_thresholds": [0.5],
            "class_labels": ["inactive", "active"],
        },
    }

    @pytest.fixture(params=_ALL_REGRESSOR_FACTORIES)
    def pretrained_regressor_path(self, request, mol_list, regression_y, tmp_path):
        """Fit a base regressor, save it, and return the save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    def test_label_encoder_preserves_thresholds(self, pretrained_regressor_path):
        """User-provided class_thresholds are preserved after construction."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        encoder = finetuner.datamodule._label_encoder
        assert encoder.params.class_thresholds[0] == [0.5]

    def test_label_encoder_preserves_class_labels(self, pretrained_regressor_path):
        """User-provided class_labels are preserved after construction."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        encoder = finetuner.datamodule._label_encoder
        assert encoder.params.class_labels[0] == ["inactive", "active"]

    def test_label_encoder_is_set(self, pretrained_regressor_path):
        """label_encoder.is_set() returns True when thresholds are configured."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        assert finetuner.datamodule._label_encoder.is_set() is True

    def test_label_encoder_is_binary_classification_type(
        self, pretrained_regressor_path
    ):
        """Label encoder is a BinaryClassificationLabelEncoder."""
        from matcha.datamodules.classic.label_encoder import (
            BinaryClassificationLabelEncoder,
        )

        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        assert isinstance(
            finetuner.datamodule._label_encoder, BinaryClassificationLabelEncoder
        )

    def test_is_classification_flag_set(self, pretrained_regressor_path):
        """datamodule.params.is_classification is True after construction."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        assert finetuner.datamodule.params.is_classification is True

    def test_predict_produces_categorical_output(
        self, pretrained_regressor_path, mol_list, regression_y
    ):
        """Predictions from a classifier with thresholds produce categorical (0/1) outputs."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0.0, 1.0})

    def test_empty_label_encoder_params_still_creates_binary_encoder(
        self, pretrained_regressor_path
    ):
        """Even without user thresholds, the encoder type is binary_classification."""
        from matcha.datamodules.classic.label_encoder import (
            BinaryClassificationLabelEncoder,
        )

        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params={},
            **FINETUNE_TRAIN,
        )
        assert isinstance(
            finetuner.datamodule._label_encoder, BinaryClassificationLabelEncoder
        )

    def test_adapt_dicts_for_mixin_sets_encoder_type(self, pretrained_regressor_path):
        """_adapt_dicts_for_mixin sets encoder_type without dropping other keys."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        # Call _adapt_dicts_for_mixin with a dict that has extra keys
        dm_dict = {
            "label_encoder_params": {
                "encoder_type": "binary_classification",
                "extra_key": "should_persist",
            },
            "is_classification": False,
        }
        result_dm, _ = finetuner._adapt_dicts_for_mixin(dm_dict, {})
        assert result_dm["is_classification"] is True
        assert (
            result_dm["label_encoder_params"]["encoder_type"] == "binary_classification"
        )
        assert result_dm["label_encoder_params"]["extra_key"] == "should_persist"


# =========================================================================
# keep_existing_predictor forwarding through sklearn wrappers (issue #85)
# =========================================================================

# Factories used to exercise the flag being set to False. The tabular MLP
# encoder does not expose the ``fp_dim`` attribute that ``keep_existing_predictor=False``
# needs at the torch layer, so we restrict the False-value forwarding tests
# to GIN and CNN modalities. The True-value tests still cover all non-Chemprop
# modalities since that path is unchanged from the pre-#85 behavior.
_FLAG_FALSE_REGRESSOR_FACTORIES = [
    pytest.param(make_gin_regressor, id="GINRegressor"),
    pytest.param(make_cnn_regressor, id="CNNRegressor"),
]

_FLAG_FALSE_CLASSIFIER_FACTORIES = [
    pytest.param(make_gin_classifier, id="GINClassifier"),
    pytest.param(make_cnn_classifier, id="CNNClassifier"),
]

_FLAG_TRUE_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLPRegressor"),
    pytest.param(make_gin_regressor, id="GINRegressor"),
    pytest.param(make_cnn_regressor, id="CNNRegressor"),
]


class TestKeepExistingPredictorForwarding:
    """Verify keep_existing_predictor is threaded through the sklearn wrappers.

    The flag exists on ``Finetuner.__init__`` but was not accepted or forwarded
    by the sklearn wrappers before issue #85. These tests exercise the plumbing
    end-to-end: signature acceptance, hparams forwarding to the underlying
    torch module, a fit+predict smoke test, and the Chemprop guard.
    """

    @pytest.fixture(params=_FLAG_FALSE_REGRESSOR_FACTORIES)
    def flag_false_regressor_path(self, request, mol_list, regression_y, tmp_path):
        """Fit a regressor for which flag=False is supported, return its save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_kep_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture(params=_FLAG_FALSE_CLASSIFIER_FACTORIES)
    def flag_false_classifier_path(self, request, mol_list, classification_y, tmp_path):
        """Fit a classifier for which flag=False is supported, return its save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_kep_{factory.__name__}")
        model.fit(mol_list, classification_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture(params=_FLAG_TRUE_REGRESSOR_FACTORIES)
    def flag_true_regressor_path(self, request, mol_list, regression_y, tmp_path):
        """Fit any non-Chemprop regressor, return its save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_kep_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture()
    def chemprop_regressor_path(self, mol_list, regression_y, tmp_path):
        """Fit a Chemprop regressor, save it, return the save path."""
        model = make_chemprop_regressor()
        save_dir = str(tmp_path / "pretrained_kep_chemprop")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    def test_regressor_construction_with_keep_existing_predictor_false(
        self, flag_false_regressor_path
    ):
        """FinetuningRegressor accepts keep_existing_predictor=False and forwards it."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=flag_false_regressor_path,
            keep_existing_predictor=False,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model.hparams["keep_existing_predictor"] is False

    def test_regressor_construction_with_keep_existing_predictor_true(
        self, flag_true_regressor_path
    ):
        """FinetuningRegressor forwards keep_existing_predictor=True (default)."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=flag_true_regressor_path,
            keep_existing_predictor=True,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model.hparams["keep_existing_predictor"] is True

    def test_classifier_construction_with_keep_existing_predictor_false(
        self, flag_false_classifier_path
    ):
        """FinetuningClassifier accepts keep_existing_predictor=False and forwards it."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=flag_false_classifier_path,
            keep_existing_predictor=False,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model.hparams["keep_existing_predictor"] is False

    def test_regressor_fit_predict_with_flag_false(
        self, mol_list, regression_y, tmp_path
    ):
        """End-to-end fit + predict works with keep_existing_predictor=False."""
        base = make_gin_regressor()
        save_dir = str(tmp_path / "pretrained_kep_smoke")
        base.fit(mol_list, regression_y)
        base.save_model(save_dir)

        finetuner = FinetuningRegressor(
            path_to_pretrained=save_dir,
            keep_existing_predictor=False,
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)

        assert preds.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(preds))

    def test_chemprop_regressor_raises_on_flag_false(self, chemprop_regressor_path):
        """Chemprop pretrained models reject keep_existing_predictor=False."""
        with pytest.raises(ValueError, match="keep_existing_predictor=False"):
            FinetuningRegressor(
                path_to_pretrained=chemprop_regressor_path,
                keep_existing_predictor=False,
                **FINETUNE_TRAIN,
            )

    def test_chemprop_regressor_accepts_flag_true(self, chemprop_regressor_path):
        """Chemprop pretrained models accept the default keep_existing_predictor=True."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=chemprop_regressor_path,
            keep_existing_predictor=True,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model is not None
