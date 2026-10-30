"""Tests for ordinary finetuner construction, prediction, and scheduling."""

import numpy as np
import pytest

from matcha.sklearn.finetuner import FinetuningClassifier, FinetuningRegressor

from .conftest import (
    FINETUNE_TRAIN,
    make_chemprop_classifier,
    make_chemprop_regressor,
    make_cnn_classifier,
    make_cnn_regressor,
    make_gin_classifier,
    make_gin_regressor,
    make_mlp_classifier,
    make_mlp_regressor,
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
# Fixtures – pretrained paths  (fresh base fit per test invocation)
# =========================================================================


@pytest.fixture(params=_REGRESSOR_FACTORIES)
def pretrained_regressor_path(request, mol_list, regression_y, tmp_path):
    """Fit and save a base regressor for one test invocation."""
    factory = request.param
    model = factory()
    save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
    model.fit(mol_list, regression_y)
    model.save_model(save_dir)
    return save_dir


@pytest.fixture(params=_CLASSIFIER_FACTORIES)
def pretrained_classifier_path(request, mol_list, classification_y, tmp_path):
    """Fit and save a base classifier for one test invocation."""
    factory = request.param
    model = factory()
    save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
    model.fit(mol_list, classification_y)
    model.save_model(save_dir)
    return save_dir


# =========================================================================
# Fixtures – fitted finetuners  (fresh fit per test invocation)
# =========================================================================


@pytest.fixture()
def fitted_finetuning_regressor(pretrained_regressor_path, mol_list, regression_y):
    """Construct and fit a FinetuningRegressor for one test invocation."""
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
    """Construct and fit a FinetuningClassifier for one test invocation."""
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
# FinetuningRegressor – fit + predict  (function-scoped fit)
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
# FinetuningClassifier – fit + predict  (function-scoped fit)
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
