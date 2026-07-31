"""Test TrainingManager through the sklearn API.

Model: AttentiveFPRegressor (graph)
Exercises: configure, build_callbacks, run (fit), is_fitted, trainer access,
    early stopping callbacks, SWA callbacks, save_checkpoint.
"""

import os
import numpy as np
import pytest

from matcha.sklearn.graph import AttentiveFPRegressor
from matcha.sklearn.managers import TrainingManager
from matcha.utils.schemas.sklearn_api import TrainingInputModel


@pytest.fixture()
def model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dims=[32],
        rwse_k=0,
        laplacian_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
        num_virtual_nodes=0,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


class TestTrainingManagerConfigure:
    """Tests for configure and params access."""

    def test_configure_creates_params(self):
        tm = TrainingManager()
        tm.configure(
            {
                "num_epochs": 5,
                "batch_size": 16,
                "accelerator": "cpu",
                "devices": 1,
                "early_stopping": False,
                "stochastic_weight_averaging": False,
                "patience": 10,
                "seed": 42,
            }
        )
        assert isinstance(tm.params, TrainingInputModel)
        assert tm.params.num_epochs == 5
        assert tm.params.batch_size == 16
        assert tm.params.seed == 42

    def test_is_fitted_false_before_training(self):
        tm = TrainingManager()
        assert tm.is_fitted is False

    def test_trainer_is_none_before_run(self):
        tm = TrainingManager()
        assert tm.trainer is None


class TestTrainingManagerCallbacks:
    """Tests for build_callbacks with various configurations."""

    def test_callbacks_without_early_stopping_or_swa(self):
        tm = TrainingManager()
        params = TrainingInputModel(
            num_epochs=1,
            batch_size=32,
            accelerator="cpu",
            devices=1,
            early_stopping=False,
            stochastic_weight_averaging=False,
            patience=10,
            seed=0,
        )
        callbacks = tm.build_callbacks(params)
        # Only LearningRateMonitor
        assert len(callbacks) == 1

    def test_callbacks_with_early_stopping(self):
        tm = TrainingManager()
        params = TrainingInputModel(
            num_epochs=10,
            batch_size=32,
            accelerator="cpu",
            devices=1,
            early_stopping=True,
            stochastic_weight_averaging=False,
            patience=5,
            seed=0,
        )
        callbacks = tm.build_callbacks(params)
        # LRMonitor + EarlyStopping + ModelCheckpoint = 3
        assert len(callbacks) == 3

    def test_callbacks_with_swa(self):
        tm = TrainingManager()
        params = TrainingInputModel(
            num_epochs=10,
            batch_size=32,
            accelerator="cpu",
            devices=1,
            early_stopping=False,
            stochastic_weight_averaging=True,
            patience=10,
            seed=0,
        )
        callbacks = tm.build_callbacks(params)
        # LRMonitor + SWA = 2
        assert len(callbacks) == 2

    def test_callbacks_with_early_stopping_and_swa(self):
        tm = TrainingManager()
        params = TrainingInputModel(
            num_epochs=10,
            batch_size=32,
            accelerator="cpu",
            devices=1,
            early_stopping=True,
            stochastic_weight_averaging=True,
            patience=5,
            seed=0,
        )
        callbacks = tm.build_callbacks(params)
        # LRMonitor + EarlyStopping + ModelCheckpoint + SWA = 4
        assert len(callbacks) == 4


class TestTrainingManagerRun:
    """Tests for the training loop via the sklearn API."""

    def test_fit_marks_model_as_fitted(self, mol_list, regression_y, model_kwargs):
        model = AttentiveFPRegressor(**model_kwargs)
        assert model.is_fitted is False
        model.fit(mol_list, regression_y)
        assert model.is_fitted is True

    def test_trainer_is_available_after_fit(self, mol_list, regression_y, model_kwargs):
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)
        assert model._training_manager.trainer is not None

    def test_predict_after_fit_returns_correct_shape(
        self, mol_list, regression_y, model_kwargs
    ):
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)
        preds = model.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(mol_list), 1)

    def test_predict_values_are_finite(self, mol_list, regression_y, model_kwargs):
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)
        preds = model.predict(mol_list)
        assert np.all(np.isfinite(preds))


class TestTrainingManagerCheckpoint:
    """Tests for save_checkpoint."""

    def test_save_checkpoint_creates_file(
        self, mol_list, regression_y, model_kwargs, tmp_path
    ):
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)
        ckpt_path = str(tmp_path / "model.ckpt")
        model._training_manager.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

    def test_save_checkpoint_raises_before_fit(self, tmp_path):
        tm = TrainingManager()
        with pytest.raises(RuntimeError, match="No trainer available"):
            tm.save_checkpoint(str(tmp_path / "model.ckpt"))
