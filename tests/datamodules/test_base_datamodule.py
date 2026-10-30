"""Tests for BaseDataModule (tested through a concrete sub-class)."""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from matcha.datamodules.classic.tabular_datamodule import TabularDataModule
from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.classic.label_encoder import (
    RegressionLabelEncoder,
    BinaryClassificationLabelEncoder,
)
from matcha.datamodules.classic.label_transform import LabelTransform


# We test BaseDataModule behaviour via TabularDataModule, the simplest
# concrete sub-class.  This avoids having to mock the abstract methods.


def _make_dm(**kwargs):
    """Helper to build a TabularDataModule with minimal defaults."""
    defaults = dict(feature_list=["ecfp"])
    defaults.update(kwargs)
    return TabularDataModule(**defaults)


# ===================================================================
# Y scaler creation
# ===================================================================


class TestCreateYScaler:
    def test_standard_scaler(self):
        dm = _make_dm(scaler_type="standard")
        assert isinstance(dm.y_scaler, StandardScaler)

    def test_quantile_scaler(self):
        dm = _make_dm(scaler_type="quantile")
        assert isinstance(dm.y_scaler, QuantileTransformer)

    def test_unknown_type_raises_error(self):
        with pytest.raises((ValueError, Exception)):
            _make_dm(scaler_type="unknown_type")


# ===================================================================
# Label encoder creation
# ===================================================================


class TestCreateLabelEncoder:
    def test_default_regression_encoder(self):
        dm = _make_dm()
        assert isinstance(dm.label_encoder, RegressionLabelEncoder)

    def test_binary_classification_encoder(self):
        dm = _make_dm(
            label_encoder_params={
                "encoder_type": "binary_classification",
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                },
            }
        )
        assert isinstance(dm.label_encoder, BinaryClassificationLabelEncoder)


# ===================================================================
# Label transform creation
# ===================================================================


class TestCreateLabelTransform:
    def test_default_label_transform(self):
        dm = _make_dm()
        assert isinstance(dm.label_transform, LabelTransform)
        assert dm.label_transform.params.transform_map is None

    def test_label_transform_with_map(self):
        dm = _make_dm(label_transform_params={"transform_map": "log10"})
        assert dm.label_transform.params.transform_map == "log10"


# ===================================================================
# Dataset property setters / getters
# ===================================================================


class TestDatasetProperties:
    def test_set_and_get_dataset_train(self, mol_list, regression_y):
        dm = _make_dm()
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        dm.dataset_train = ds
        assert dm.dataset_train is ds

    def test_set_invalid_dataset_raises(self):
        dm = _make_dm()
        with pytest.raises(ValueError):
            dm.dataset_train = "not_a_dataset"

    def test_dataset_val_none_by_default(self):
        dm = _make_dm()
        assert dm.dataset_val is None

    def test_dataset_test_none_by_default(self):
        dm = _make_dm()
        assert dm.dataset_test is None

    def test_dataset_predict_none_by_default(self):
        dm = _make_dm()
        assert dm.dataset_predict is None


# ===================================================================
# _handle_empty_y
# ===================================================================


class TestHandleEmptyY:
    def test_returns_404_array(self):
        dm = _make_dm()
        y = dm._handle_empty_y(10)
        assert y.shape == (10, 1)
        assert (y == 404).all()

    def test_uses_fitted_n_tasks(self, mol_list, regression_y):
        dm = _make_dm()
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        y = dm._handle_empty_y(5)
        assert y.shape == (5, 1)


# ===================================================================
# _guess_n_jobs
# ===================================================================


class TestGuessNJobs:
    def test_small_dataset_returns_1(self):
        dm = _make_dm()
        n = dm._guess_n_jobs(list(range(10)))
        assert n == 1

    def test_medium_dataset(self):
        dm = _make_dm()
        n = dm._guess_n_jobs(list(range(5000)))
        assert n >= 1


# ===================================================================
# Y scaling (fit / transform / invert)
# ===================================================================


class TestYScaling:
    def test_fit_y_sets_scaler(self, mol_list, regression_y):
        dm = _make_dm()
        ds = dm.generate_features(mol_list, regression_y, n_jobs=1)
        dm._fit_y(ds)
        assert hasattr(dm.y_scaler, "mean_")

    def test_transform_y_changes_values(self, mol_list, regression_y):
        dm = _make_dm()
        ds = dm.generate_features(mol_list, regression_y, n_jobs=1)
        y_before = ds.datasets["y"].clone()
        dm._fit_y(ds)
        dm._transform_y(ds)
        y_after = ds.datasets["y"]
        assert not torch.allclose(y_before, y_after)

    def test_invert_y_recovers_original(self, mol_list, regression_y):
        dm = _make_dm()
        ds = dm.generate_features(mol_list, regression_y, n_jobs=1)
        y_original = ds.datasets["y"].clone()
        dm._fit_y(ds)
        dm._transform_y(ds)
        dm._invert_y(ds)
        torch.testing.assert_close(ds.datasets["y"], y_original, atol=1e-4, rtol=1e-4)

    def test_classification_skips_y_scaling(self, mol_list, classification_y):
        dm = _make_dm(is_classification=True)
        ds = dm.generate_features(mol_list, classification_y, n_jobs=1)
        y_before = ds.datasets["y"].clone()
        dm._fit_y(ds)
        dm._transform_y(ds)
        # Classification should not change y
        torch.testing.assert_close(ds.datasets["y"], y_before)


# ===================================================================
# invert_y (numpy pathway)
# ===================================================================


class TestInvertYNumpy:
    def test_invert_y_numpy(self, mol_list, regression_y):
        dm = _make_dm()
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        scaled = dm.y_scaler.transform(regression_y)
        inverted = dm.invert_y(scaled)
        np.testing.assert_allclose(inverted, regression_y, atol=1e-4)


# ===================================================================
# _process_y (bound mask handling)
# ===================================================================


class TestProcessY:
    def test_bound_mask_stacks_censor_dim(
        self, mol_list, regression_y, bound_mask_mixed
    ):
        dm = _make_dm()
        ds = dm.generate_features(mol_list, regression_y, n_jobs=1)
        dm._process_y(ds, bound_mask_mixed)
        # y should now be (N, tasks, 2) – value + mask
        assert ds.datasets["y"].ndim == 3
        assert ds.datasets["y"].shape[2] == 2

    def test_no_bound_mask_keeps_y_unchanged(self, mol_list, regression_y):
        dm = _make_dm()
        ds = dm.generate_features(mol_list, regression_y, n_jobs=1)
        y_before = ds.datasets["y"].clone()
        dm._process_y(ds, bound_mask=None)
        torch.testing.assert_close(ds.datasets["y"], y_before)


# ===================================================================
# setup + dataloaders
# ===================================================================


class TestSetup:
    def test_setup_fit_creates_train_loader(self, mol_list, regression_y):
        dm = _make_dm(batch_size=8)
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        dm.dataset_train = ds
        dm.setup("fit")
        assert dm.train_dataloader() is not None

    def test_setup_fit_with_val(self, mol_list, regression_y):
        dm = _make_dm(batch_size=8)
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        dm.dataset_train = ds
        dm.dataset_val = ds
        dm.setup("fit")
        assert dm.val_dataloader() is not None

    def test_setup_test(self, mol_list, regression_y):
        dm = _make_dm(batch_size=8)
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        dm.dataset_test = ds
        dm.setup("test")
        assert dm.test_dataloader() is not None

    def test_setup_predict(self, mol_list, regression_y):
        dm = _make_dm(batch_size=8)
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        dm.dataset_predict = ds
        dm.setup("predict")
        assert dm.predict_dataloader() is not None


# ===================================================================
# collate_fn
# ===================================================================


class TestCollateFn:
    def test_collate_fn_returns_dict(self, mol_list, regression_y):
        dm = _make_dm(batch_size=4)
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        batch = [ds[i] for i in range(4)]
        collated = dm.collate_fn(batch)
        assert isinstance(collated, dict)
        assert "mol_features" in collated
        assert "y" in collated


# ===================================================================
# configure_label_encoder / parse_output
# ===================================================================


class TestConfigureLabelEncoder:
    def test_configure_and_parse(self, mol_list, regression_y):
        dm = _make_dm()
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        dm.configure_label_encoder(
            {0: {"task_label": "logS", "class_thresholds": None, "class_labels": None}}
        )
        preds = np.random.rand(10, 1)
        result = dm.parse_output(preds, "preds")
        assert "logS_preds" in result.columns

    def test_has_class_labels_false_by_default(self):
        dm = _make_dm()
        assert dm.has_class_labels() is False


# ===================================================================
# DataModuleRegistry
# ===================================================================


class TestDataModuleRegistry:
    def test_tabular_in_registry(self):
        assert "tabular" in DataModuleRegistry

    def test_graph_in_registry(self):
        assert "graph" in DataModuleRegistry

    def test_clm_in_registry(self):
        assert "clm" in DataModuleRegistry

    def test_combined_in_registry(self):
        assert "combined" in DataModuleRegistry

    def test_chemprop_in_registry(self):
        assert "chemprop" in DataModuleRegistry
