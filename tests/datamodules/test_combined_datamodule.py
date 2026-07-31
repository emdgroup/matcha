"""Tests for CombinedDataModule."""

import numpy as np

from matcha.datamodules.classic.combined_datamodule import (
    CombinedDataModule,
    default_merge,
)
from matcha.datamodules.classic.tabular_datamodule import TabularDataModule
from matcha.datamodules.classic.graph_datamodule import GraphDataModule
from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.utils import CombinedStackDataset


# ===================================================================
# Construction
# ===================================================================


class TestCombinedDataModuleInit:
    def test_default_construction(self):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        assert dm.params.datamodule_type == "combined"
        assert len(dm.datamodules) == 2

    def test_registry_has_combined(self):
        assert "combined" in DataModuleRegistry

    def test_tabular_datamodule_idx_detected(self):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        assert dm.tabular_datamodule_idx == 0


# ===================================================================
# Default merge
# ===================================================================


class TestDefaultMerge:
    def test_returns_first_element(self):
        result = default_merge([1, 2, 3])
        assert result == 1


# ===================================================================
# generate_features
# ===================================================================


class TestCombinedGenerateFeatures:
    def test_returns_combined_stack_dataset(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        ds = dm.generate_features(small_mol_list, small_regression_y, n_jobs=1)
        assert isinstance(ds, CombinedStackDataset)

    def test_combined_dataset_has_all_keys(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        ds = dm.generate_features(small_mol_list, small_regression_y, n_jobs=1)
        item = ds[0]
        assert "mol_features" in item
        assert "graph" in item
        assert "y" in item


# ===================================================================
# featurize
# ===================================================================


class TestCombinedFeaturize:
    def test_featurize_returns_combined_stack_dataset(
        self, small_mol_list, small_regression_y
    ):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert isinstance(ds, CombinedStackDataset)

    def test_featurize_y_shape(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        # y is stored in each sub-dataset – check from item access
        item = ds[0]
        assert "y" in item


class TestCombinedFeaturizeTest:
    def test_featurize_test_mode(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        ds_test = dm.featurize(
            small_mol_list[:3], small_regression_y[:3], is_training=False, n_jobs=1
        )
        assert len(ds_test) == 3


# ===================================================================
# Scaling delegation
# ===================================================================


class TestCombinedScaling:
    def test_fit_delegates_to_sub_datamodules(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        # After featurize, the tabular sub-module should have fitted scalers
        assert hasattr(tab.x_scaler, "mean_")


# ===================================================================
# Collate function
# ===================================================================


class TestCombinedCollate:
    def test_collate_fn_produces_dict(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph], batch_size=4)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        # Manually collate a small batch
        batch = [ds[i] for i in range(min(3, len(ds)))]
        collated = dm.collate_fn(batch)
        assert isinstance(collated, dict)
        assert "mol_features" in collated
        assert "graph" in collated
        assert "y" in collated


# ===================================================================
# invert_y delegation
# ===================================================================


class TestCombinedInvertY:
    def test_invert_y_delegates_to_first(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        dummy_y = np.zeros((3, 1))
        result = dm.invert_y(dummy_y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)


# ===================================================================
# State dict
# ===================================================================


class TestCombinedStateDict:
    def test_state_dict_keys(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()
        assert "ID" in sd
        assert sd["ID"] == "combined"
        assert "params" in sd
        assert "keys_collate" in sd

    def test_load_state_dict_roundtrip(self, small_mol_list, small_regression_y):
        tab = TabularDataModule(feature_list=["ecfp"])
        graph = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm = CombinedDataModule(datamodules=[tab, graph])
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()

        dm2 = CombinedDataModule.dummy()
        dm2.load_state_dict(sd)
        assert len(dm2.datamodules) == 2


# ===================================================================
# Dummy
# ===================================================================


class TestCombinedDummy:
    def test_dummy_creation(self):
        dm = CombinedDataModule.dummy()
        assert isinstance(dm, CombinedDataModule)
        assert len(dm.datamodules) == 2
