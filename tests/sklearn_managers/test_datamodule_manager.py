"""Test DataModuleManager through the sklearn API.

Model: GINRegressor (graph), ChempropRegressor (chemprop)
Exercises: featurize, set_train_dataset, set_val_dataset, set_predict_dataset,
    set_batch_size, state_dict / load_state_dict, prepare_fit_datasets,
    invert_y, setup, predict_dataloader.
"""

import numpy as np
import pytest
from rdkit.Chem.rdchem import Mol

from matcha.sklearn.graph import ChempropRegressor, GINRegressor


# =========================================================================
# GIN kwargs / fixtures
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


@pytest.fixture()
def fitted_model(mol_list: list[Mol], regression_y, model_kwargs):
    model = GINRegressor(**model_kwargs)
    model.fit(mol_list, regression_y)
    return model


# =========================================================================
# Chemprop kwargs / fixtures
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


@pytest.fixture()
def chemprop_fitted_model(mol_list: list[Mol], regression_y, chemprop_model_kwargs):
    model = ChempropRegressor(**chemprop_model_kwargs)
    model.fit(mol_list, regression_y)
    return model


class TestDataModuleManagerFeaturize:
    """Tests for the featurize / transform path."""

    def test_transform_returns_stack_dataset(
        self, mol_list, regression_y, model_kwargs
    ):
        model = GINRegressor(**model_kwargs)
        dataset = model.transform(mol_list, regression_y)
        # StackDataset or CombinedStackDataset should be iterable
        assert len(dataset) == len(mol_list)

    def test_transform_inference_mode_without_y(self, fitted_model, mol_list):
        dataset = fitted_model.transform(mol_list, y=None, is_training=False)
        assert len(dataset) == len(mol_list)


class TestDataModuleManagerDatasetAssignment:
    """Tests for set_*_dataset and set_batch_size."""

    def test_set_batch_size(self, mol_list, regression_y, model_kwargs):
        model = GINRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)
        model._datamodule_manager.set_batch_size(16)
        assert model.datamodule.params.batch_size == 16

    def test_set_predict_dataset(self, fitted_model, mol_list):
        dataset = fitted_model.transform(mol_list, y=None, is_training=False)
        fitted_model._datamodule_manager.set_predict_dataset(dataset)
        assert fitted_model.datamodule.dataset_predict is dataset


class TestDataModuleManagerStateDict:
    """Tests for state_dict / load_state_dict round-trip."""

    def test_state_dict_is_dict(self, fitted_model):
        state = fitted_model._datamodule_manager.state_dict()
        assert isinstance(state, dict)

    def test_state_dict_contains_id(self, fitted_model):
        state = fitted_model._datamodule_manager.state_dict()
        assert "ID" in state

    def test_load_state_dict_round_trip(self, fitted_model, mol_list, regression_y):
        state = fitted_model._datamodule_manager.state_dict()
        # Predict before reload
        preds_before = fitted_model.predict(mol_list)

        # Reload state
        fitted_model._datamodule_manager.load_state_dict(state)

        preds_after = fitted_model.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestDataModuleManagerInvertY:
    """Tests for invert_y (undo label scaling)."""

    def test_invert_y_returns_ndarray(self, fitted_model):
        dummy_preds = np.random.randn(5, 1).astype(np.float32)
        inverted = fitted_model._datamodule_manager.invert_y(dummy_preds)
        assert isinstance(inverted, np.ndarray)
        assert inverted.shape == dummy_preds.shape


class TestDataModuleManagerPrepare:
    """Tests for prepare_fit_datasets with molecule list input."""

    def test_prepare_fit_sets_train_dataset(self, mol_list, regression_y, model_kwargs):
        model = GINRegressor(**model_kwargs)
        model._datamodule_manager.prepare_fit_datasets(
            x=mol_list,
            y=regression_y,
            bound_mask=None,
            validation_set=None,
            transform_fn=model.transform,
            early_stopping=False,
            seed=0,
            batch_size=32,
        )
        assert model.datamodule.dataset_train is not None


# =========================================================================
# Chemprop variants – exercises the ChempropDataModule code path
# =========================================================================


class TestChempropDataModuleManagerFeaturize:
    """Tests for the featurize / transform path with ChempropRegressor."""

    def test_transform_returns_stack_dataset(
        self, mol_list, regression_y, chemprop_model_kwargs
    ):
        model = ChempropRegressor(**chemprop_model_kwargs)
        dataset = model.transform(mol_list, regression_y)
        assert len(dataset) == len(mol_list)

    def test_transform_inference_mode_without_y(self, chemprop_fitted_model, mol_list):
        dataset = chemprop_fitted_model.transform(mol_list, y=None, is_training=False)
        assert len(dataset) == len(mol_list)


class TestChempropDataModuleManagerDatasetAssignment:
    """Tests for set_*_dataset and set_batch_size with ChempropRegressor."""

    def test_set_batch_size(self, mol_list, regression_y, chemprop_model_kwargs):
        model = ChempropRegressor(**chemprop_model_kwargs)
        model.fit(mol_list, regression_y)
        model._datamodule_manager.set_batch_size(16)
        assert model.datamodule.params.batch_size == 16

    def test_set_predict_dataset(self, chemprop_fitted_model, mol_list):
        dataset = chemprop_fitted_model.transform(mol_list, y=None, is_training=False)
        chemprop_fitted_model._datamodule_manager.set_predict_dataset(dataset)
        assert chemprop_fitted_model.datamodule.dataset_predict is dataset


class TestChempropDataModuleManagerStateDict:
    """Tests for state_dict / load_state_dict round-trip with ChempropRegressor."""

    def test_state_dict_is_dict(self, chemprop_fitted_model):
        state = chemprop_fitted_model._datamodule_manager.state_dict()
        assert isinstance(state, dict)

    def test_state_dict_contains_id(self, chemprop_fitted_model):
        state = chemprop_fitted_model._datamodule_manager.state_dict()
        assert "ID" in state

    def test_load_state_dict_round_trip(
        self, chemprop_fitted_model, mol_list, regression_y
    ):
        state = chemprop_fitted_model._datamodule_manager.state_dict()
        preds_before = chemprop_fitted_model.predict(mol_list)

        chemprop_fitted_model._datamodule_manager.load_state_dict(state)

        preds_after = chemprop_fitted_model.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestChempropDataModuleManagerInvertY:
    """Tests for invert_y (undo label scaling) with ChempropRegressor."""

    def test_invert_y_returns_ndarray(self, chemprop_fitted_model):
        dummy_preds = np.random.randn(5, 1).astype(np.float32)
        inverted = chemprop_fitted_model._datamodule_manager.invert_y(dummy_preds)
        assert isinstance(inverted, np.ndarray)
        assert inverted.shape == dummy_preds.shape


class TestChempropDataModuleManagerPrepare:
    """Tests for prepare_fit_datasets with ChempropRegressor."""

    def test_prepare_fit_sets_train_dataset(
        self, mol_list, regression_y, chemprop_model_kwargs
    ):
        model = ChempropRegressor(**chemprop_model_kwargs)
        model._datamodule_manager.prepare_fit_datasets(
            x=mol_list,
            y=regression_y,
            bound_mask=None,
            validation_set=None,
            transform_fn=model.transform,
            early_stopping=False,
            seed=0,
            batch_size=32,
        )
        assert model.datamodule.dataset_train is not None
