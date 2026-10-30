"""Test autoload for all modalities (tabular, graph, CLM, chemprop).

Exercises: save a model → autoload it from the folder → verify predictions
match.  Also tests autoload on ensemble artifacts.

Each modality is tested with both a regressor and a classifier.

Runtime optimisation: expensive ``fit`` calls are performed once per factory
via fixtures; multiple assertions are run against the same saved artefact.
"""

import os

import numpy as np
import pytest

from matcha.sklearn import Ensemble
from matcha.sklearn.autoload import autoload

from .conftest import (
    make_cnn_regressor,
    make_cnn_classifier,
    N_MODELS,
)


# =========================================================================
# Local factory overrides for CNN (num_augmentations=0 for autoload)
# =========================================================================
# Autoload tests use num_augmentations=0 for CNN to avoid augmentation
# mismatches on reload.  Other modalities use the conftest factories as-is.

_AUTOLOAD_REGRESSOR_FACTORIES = [
    pytest.param("mlp", id="MLPRegressor"),
    pytest.param("gin", id="GINRegressor"),
    pytest.param("cnn", id="CNNRegressor"),
    pytest.param("chemprop", id="ChempropRegressor"),
]

_AUTOLOAD_CLASSIFIER_FACTORIES = [
    pytest.param("mlp", id="MLPClassifier"),
    pytest.param("gin", id="GINClassifier"),
    pytest.param("cnn", id="CNNClassifier"),
    pytest.param("chemprop", id="ChempropClassifier"),
]


def _get_regressor(key):
    from .conftest import (
        make_mlp_regressor,
        make_gin_regressor,
        make_chemprop_regressor,
    )

    _map = {
        "mlp": make_mlp_regressor,
        "gin": make_gin_regressor,
        "cnn": lambda: make_cnn_regressor(num_augmentations=0),
        "chemprop": make_chemprop_regressor,
    }
    return _map[key]()


def _get_classifier(key):
    from .conftest import (
        make_mlp_classifier,
        make_gin_classifier,
        make_chemprop_classifier,
    )

    _map = {
        "mlp": make_mlp_classifier,
        "gin": make_gin_classifier,
        "cnn": lambda: make_cnn_classifier(num_augmentations=0),
        "chemprop": make_chemprop_classifier,
    }
    return _map[key]()


# =========================================================================
# Fixtures – fit once, save once, reuse for all assertions
# =========================================================================


@pytest.fixture(params=_AUTOLOAD_REGRESSOR_FACTORIES)
def saved_regressor(request, mol_list, regression_y, tmp_path):
    """Fit a regressor, save it, return (save_dir, original_preds, mol_list)."""
    model = _get_regressor(request.param)
    model.fit(mol_list, regression_y)
    preds_original = model.predict(mol_list)
    save_dir = str(tmp_path / "saved_model")
    model.save_model(save_dir)
    return save_dir, preds_original, mol_list


@pytest.fixture(params=_AUTOLOAD_CLASSIFIER_FACTORIES)
def saved_classifier(request, mol_list, classification_y, tmp_path):
    """Fit a classifier, save it, return (save_dir, original_preds, mol_list)."""
    model = _get_classifier(request.param)
    model.fit(mol_list, classification_y)
    preds_original = model.predict(mol_list)
    save_dir = str(tmp_path / "saved_model")
    model.save_model(save_dir)
    return save_dir, preds_original, mol_list


@pytest.fixture(params=_AUTOLOAD_REGRESSOR_FACTORIES)
def saved_ens_regressor(request, mol_list, regression_y, tmp_path):
    """Fit an ensemble regressor, save it, return (save_dir, mol_list)."""
    template = _get_regressor(request.param)
    ens = Ensemble(model=template, n_models=N_MODELS)
    ens.fit(mol_list, regression_y)
    save_dir = str(tmp_path / "ens")
    ens.save_model(save_dir)
    return save_dir, mol_list


@pytest.fixture(params=_AUTOLOAD_CLASSIFIER_FACTORIES)
def saved_ens_classifier(request, mol_list, classification_y, tmp_path):
    """Fit an ensemble classifier, save it, return (save_dir, mol_list)."""
    template = _get_classifier(request.param)
    ens = Ensemble(model=template, n_models=N_MODELS)
    ens.fit(mol_list, classification_y)
    save_dir = str(tmp_path / "ens")
    ens.save_model(save_dir)
    return save_dir, mol_list


# =========================================================================
# Autoload single-model regressor tests  (one fit per factory)
# =========================================================================


class TestAutoloadRegressor:
    """Test autoload for single regressor models (save → autoload → predict)."""

    def test_autoload_returns_fitted_model(self, saved_regressor):
        save_dir, _, _ = saved_regressor
        loaded = autoload(save_dir, accelerator="cpu")
        assert loaded.is_fitted is True

    def test_autoload_predicts_same_shape(self, saved_regressor):
        save_dir, preds_original, mol_list = saved_regressor
        loaded = autoload(save_dir, accelerator="cpu")
        preds = loaded.predict(mol_list)
        assert preds.shape == preds_original.shape

    def test_autoload_predicts_similar_values(self, saved_regressor):
        save_dir, preds_original, mol_list = saved_regressor
        loaded = autoload(save_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=3)


# =========================================================================
# Autoload single-model classifier tests  (one fit per factory)
# =========================================================================


class TestAutoloadClassifier:
    """Test autoload for single classifier models (save → autoload → predict)."""

    def test_autoload_returns_fitted_model(self, saved_classifier):
        save_dir, _, _ = saved_classifier
        loaded = autoload(save_dir, accelerator="cpu")
        assert loaded.is_fitted is True

    def test_autoload_predicts_same_shape(self, saved_classifier):
        save_dir, preds_original, mol_list = saved_classifier
        loaded = autoload(save_dir, accelerator="cpu")
        preds = loaded.predict(mol_list)
        assert preds.shape == preds_original.shape

    def test_autoload_predicts_similar_values(self, saved_classifier):
        save_dir, preds_original, mol_list = saved_classifier
        loaded = autoload(save_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=3)


# =========================================================================
# Autoload ensemble regressor tests  (one fit per factory)
# =========================================================================


class TestAutoloadEnsembleRegressor:
    """Test autoload for ensemble regressor models (save → autoload → predict)."""

    def test_autoload_ensemble_returns_ensemble(self, saved_ens_regressor):
        save_dir, _ = saved_ens_regressor
        loaded = autoload(save_dir, accelerator="cpu")
        assert isinstance(loaded, Ensemble)

    def test_autoload_ensemble_predicts(self, saved_ens_regressor):
        save_dir, mol_list = saved_ens_regressor
        loaded = autoload(save_dir, accelerator="cpu")
        mean, std = loaded.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(mean))


# =========================================================================
# Autoload ensemble classifier tests  (one fit per factory)
# =========================================================================


class TestAutoloadEnsembleClassifier:
    """Test autoload for ensemble classifier models (save → autoload → predict)."""

    def test_autoload_ensemble_returns_ensemble(self, saved_ens_classifier):
        save_dir, _ = saved_ens_classifier
        loaded = autoload(save_dir, accelerator="cpu")
        assert isinstance(loaded, Ensemble)

    def test_autoload_ensemble_predicts(self, saved_ens_classifier):
        save_dir, mol_list = saved_ens_classifier
        loaded = autoload(save_dir, accelerator="cpu")
        mean, std = loaded.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(mean))


# =========================================================================
# Autoload error handling
# =========================================================================


class TestAutoloadErrors:
    """Test that autoload raises informative errors for invalid paths."""

    def test_autoload_raises_on_empty_folder(self, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError, match="No recognized serialization"):
            autoload(empty_dir)

    def test_autoload_raises_on_nonexistent_path(self, tmp_path):
        bogus_path = str(tmp_path / "does_not_exist")
        with pytest.raises((FileNotFoundError, OSError)):
            autoload(bogus_path)
