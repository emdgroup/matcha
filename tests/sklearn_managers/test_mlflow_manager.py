"""Test MLFlowManager through the sklearn API.

Model: GatedGCNRegressor (graph), ChempropClassifier (chemprop)
Exercises: setup_experiment, is_active, create_logger, log_training (end-to-end
    with set_mlflow_experiment + fit).
"""

import os
import shutil

import pytest

from matcha.sklearn.graph import ChempropClassifier, GatedGCNRegressor
from matcha.sklearn.managers import MLFlowManager
from matcha.utils.schemas.sklearn_api import MLFlowInputModel


# =========================================================================
# GatedGCN kwargs
# =========================================================================


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


# =========================================================================
# Chemprop kwargs
# =========================================================================


@pytest.fixture()
def chemprop_model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dim=32,
        pred_num_layers=1,
        feature_list=None,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


class TestMLFlowManagerSetup:
    """Tests for setup_experiment and is_active."""

    def test_is_active_false_by_default(self):
        mgr = MLFlowManager()
        assert mgr.is_active is False

    def test_setup_experiment_activates(self):
        mgr = MLFlowManager()
        result = mgr.setup_experiment(
            experiment_name="test_exp",
            run_name="test_run",
        )
        assert mgr.is_active is True
        assert isinstance(result, MLFlowInputModel)
        assert result.experiment == "test_exp"
        assert result.run == "test_run"

    def test_setup_experiment_with_tag(self):
        mgr = MLFlowManager()
        mgr.setup_experiment(
            experiment_name="test_exp",
            run_name="test_run",
            tag={"version": "1.0"},
        )
        assert mgr.params.tag == {"version": "1.0"}

    def test_setup_experiment_default_log_dir(self):
        mgr = MLFlowManager()
        mgr.setup_experiment(experiment_name="exp")
        assert mgr.params.log_dir == "./matcha_log"

    def test_setup_experiment_custom_log_dir(self):
        mgr = MLFlowManager()
        mgr.setup_experiment(experiment_name="exp", log_dir="/tmp/custom_log")
        assert mgr.params.log_dir == "/tmp/custom_log"


class TestMLFlowManagerCreateLogger:
    """Tests for create_logger."""

    def test_create_logger_returns_matcha_logger(self):
        mgr = MLFlowManager()
        mgr.setup_experiment(
            experiment_name="test_exp",
            run_name="test_run",
            log_dir="/tmp/mlflow_test_manager",
        )
        logger = mgr.create_logger()
        # MatchaLogger should have experiment_name attribute
        assert logger is not None
        # Clean up
        if os.path.exists("/tmp/mlflow_test_manager"):
            shutil.rmtree("/tmp/mlflow_test_manager")


class TestMLFlowManagerViaSklearn:
    """Tests for MLFlow integration through the sklearn model API."""

    def test_set_mlflow_experiment_activates_manager(self, model_kwargs):
        model = GatedGCNRegressor(**model_kwargs)
        assert model._mlflow_manager.is_active is False
        model.set_mlflow_experiment(
            experiment_name="sklearn_test",
            run_name="run_1",
        )
        assert model._mlflow_manager.is_active is True

    def test_fit_with_mlflow_creates_artifacts(
        self, mol_list, regression_y, model_kwargs, tmp_path
    ):
        model = GatedGCNRegressor(**model_kwargs)
        log_dir = str(tmp_path / "mlflow_log")
        model.set_mlflow_experiment(
            experiment_name="manager_test",
            run_name="fit_test",
            log_dir=log_dir,
        )
        model.fit(mol_list, regression_y)
        # MLflow should have created something in the log directory
        assert os.path.exists(log_dir)
        assert model.is_fitted is True


# =========================================================================
# Chemprop variants – exercises ChempropClassifier through the MLFlow path
# =========================================================================


class TestChempropMLFlowManagerViaSklearn:
    """Tests for MLFlow integration through the ChempropClassifier sklearn API."""

    def test_set_mlflow_experiment_activates_manager(self, chemprop_model_kwargs):
        model = ChempropClassifier(**chemprop_model_kwargs)
        assert model._mlflow_manager.is_active is False
        model.set_mlflow_experiment(
            experiment_name="sklearn_chemprop_test",
            run_name="run_1",
        )
        assert model._mlflow_manager.is_active is True

    def test_fit_with_mlflow_creates_artifacts(
        self, mol_list, classification_y, chemprop_model_kwargs, tmp_path
    ):
        model = ChempropClassifier(**chemprop_model_kwargs)
        log_dir = str(tmp_path / "mlflow_log")
        model.set_mlflow_experiment(
            experiment_name="chemprop_manager_test",
            run_name="fit_test",
            log_dir=log_dir,
        )
        model.fit(mol_list, classification_y)
        # MLflow should have created something in the log directory
        assert os.path.exists(log_dir)
        assert model.is_fitted is True
