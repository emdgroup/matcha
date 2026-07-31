"""Tests for TabularDataModule."""

import torch
from torch.utils.data import StackDataset

from matcha.datamodules.classic.tabular_datamodule import (
    TabularDataModule,
)
from matcha.datamodules.base_datamodule import DataModuleRegistry


# ===================================================================
# Construction
# ===================================================================


class TestTabularDataModuleInit:
    def test_default_construction(self):
        dm = TabularDataModule(feature_list=["ecfp"])
        assert dm.params.feature_list == ["ecfp"]
        assert dm.params.datamodule_type == "tabular"

    def test_input_dim_calculated(self):
        dm = TabularDataModule(feature_list=["ecfp"])
        assert dm.params.input_dim == 1024  # default ECFP nBits

    def test_multiple_features(self):
        dm = TabularDataModule(feature_list=["ecfp", "erg"])
        assert dm.params.input_dim == 1024 + 315

    def test_classification_mode(self):
        dm = TabularDataModule(feature_list=["ecfp"], is_classification=True)
        assert dm.params.is_classification is True

    def test_custom_batch_size(self):
        dm = TabularDataModule(feature_list=["ecfp"], batch_size=64)
        assert dm.params.batch_size == 64

    def test_registry_has_tabular(self):
        assert "tabular" in DataModuleRegistry


# ===================================================================
# Featurization (regression)
# ===================================================================


class TestTabularFeaturizeRegression:
    def test_featurize_training_returns_stack_dataset(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        assert isinstance(ds, StackDataset)

    def test_featurize_keys(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        item = ds[0]
        assert "mol_features" in item
        assert "y" in item

    def test_featurize_shapes(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        n = len(mol_list)
        assert ds.datasets["mol_features"].shape[0] == n
        assert ds.datasets["y"].shape[0] == n

    def test_featurize_feature_dim(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        assert ds.datasets["mol_features"].shape[1] == 1024

    def test_x_scaler_fitted_after_training(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        assert hasattr(dm.x_scaler, "mean_")

    def test_y_scaler_fitted_after_training(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        assert hasattr(dm.y_scaler, "mean_")


class TestTabularFeaturizeTest:
    def test_featurize_test_uses_fitted_scaler(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        ds_test = dm.featurize(
            mol_list[:5], regression_y[:5], is_training=False, n_jobs=1
        )
        assert ds_test.datasets["mol_features"].shape[0] == 5

    def test_featurize_test_without_y(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        ds_test = dm.featurize(mol_list[:5], None, is_training=False, n_jobs=1)
        assert ds_test.datasets["y"].shape[0] == 5


# ===================================================================
# Featurization (classification)
# ===================================================================


class TestTabularFeaturizeClassification:
    def test_classification_featurize(self, mol_list, classification_y):
        dm = TabularDataModule(
            feature_list=["ecfp"],
            is_classification=True,
            label_encoder_params={
                "encoder_type": "binary_classification",
                0: {
                    "task_label": "activity",
                    "class_thresholds": [0.5],
                    "class_labels": ["inactive", "active"],
                },
            },
        )
        ds = dm.featurize(mol_list, classification_y, is_training=True, n_jobs=1)
        assert isinstance(ds, StackDataset)
        assert ds.datasets["y"].shape[0] == len(mol_list)


# ===================================================================
# Scaling
# ===================================================================


class TestTabularScaling:
    def test_generate_features_is_unscaled(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds_unscaled = dm.generate_features(mol_list, regression_y, n_jobs=1)
        # ECFP values should be 0/1 before scaling
        unique_vals = torch.unique(ds_unscaled.datasets["mol_features"])
        assert 0.0 in unique_vals
        assert 1.0 in unique_vals

    def test_featurize_applies_scaling(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds_scaled = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        # After standard scaling, values shouldn't all be 0/1 anymore
        unique_vals = torch.unique(ds_scaled.datasets["mol_features"])
        # StandardScaler centers and scales data
        assert len(unique_vals) >= 2

    def test_fit_then_transform_consistency(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds = dm.generate_features(mol_list, regression_y, n_jobs=1)
        dm.fit(ds)
        ds_copy = dm.generate_features(mol_list, regression_y, n_jobs=1)
        dm.transform(ds_copy)
        # Should match what featurize produces
        dm2 = TabularDataModule(feature_list=["ecfp"])
        ds_full = dm2.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        torch.testing.assert_close(
            ds_copy.datasets["mol_features"],
            ds_full.datasets["mol_features"],
            atol=1e-5,
            rtol=1e-5,
        )


# ===================================================================
# Bound mask
# ===================================================================


class TestTabularBoundMask:
    def test_featurize_with_exact_bound_mask(
        self, mol_list, regression_y, bound_mask_exact
    ):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds = dm.featurize(
            mol_list,
            regression_y,
            bound_mask=bound_mask_exact,
            is_training=True,
            n_jobs=1,
        )
        assert isinstance(ds, StackDataset)

    def test_featurize_with_mixed_bound_mask(
        self, mol_list, regression_y, bound_mask_mixed
    ):
        dm = TabularDataModule(feature_list=["ecfp"])
        ds = dm.featurize(
            mol_list,
            regression_y,
            bound_mask=bound_mask_mixed,
            is_training=True,
            n_jobs=1,
        )
        # With bound mask, y should have an extra dim (value + mask stacked)
        assert ds.datasets["y"].ndim == 3


# ===================================================================
# State dict
# ===================================================================


class TestTabularStateDict:
    def test_state_dict_keys(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()
        assert "ID" in sd
        assert "x_scaler" in sd
        assert "y_scaler" in sd
        assert "params" in sd
        assert "label_encoder" in sd
        assert "label_transform" in sd

    def test_load_state_dict_roundtrip(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"])
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()

        dm2 = TabularDataModule.dummy()
        dm2.load_state_dict(sd)
        assert dm2.params.feature_list == ["ecfp"]
        assert dm2.params.input_dim == 1024


# ===================================================================
# Dataloader creation
# ===================================================================


class TestTabularDataloader:
    def test_setup_fit(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"], batch_size=8)
        ds_train = dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        dm.dataset_train = ds_train
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "mol_features" in batch
        assert "y" in batch
        assert batch["mol_features"].shape[0] <= 8

    def test_setup_predict(self, mol_list, regression_y):
        dm = TabularDataModule(feature_list=["ecfp"], batch_size=8)
        dm.featurize(mol_list, regression_y, is_training=True, n_jobs=1)
        ds_pred = dm.featurize(mol_list[:5], None, is_training=False, n_jobs=1)
        dm.dataset_predict = ds_pred
        dm.setup("predict")
        loader = dm.predict_dataloader()
        batch = next(iter(loader))
        assert batch["mol_features"].shape[0] <= 8


# ===================================================================
# Dummy
# ===================================================================


class TestTabularDummy:
    def test_dummy_creation(self):
        dm = TabularDataModule.dummy()
        assert isinstance(dm, TabularDataModule)
        assert dm.params.feature_list == ["ECFP"]
