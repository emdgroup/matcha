"""Tests for offline graph pretraining label scaling."""

import numpy as np
import torch

from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)


class TestFeaturize:
    def test_featurize_no_scaling_by_default(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        y_out = ds.datasets["y_graph"].numpy()
        # With scale_y_graph=False, values should be unchanged
        np.testing.assert_allclose(y_out, y_graph_small.astype(np.float32), rtol=1e-5)

    def test_featurize_with_scaling(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            scale_y_graph=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        y_out = ds.datasets["y_graph"].numpy()
        # After standard scaling the mean should be ~0
        assert abs(y_out.mean()) < 0.5

    def test_featurize_test_mode_uses_fitted_scaler(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_graph=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        # Fit
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        # Transform using the fitted scaler
        ds_test = dm.featurize(
            small_mol_list[:2],
            y_graph_small[:2],
            y_node_small[:2],
            is_training=False,
            n_jobs=1,
        )
        assert ds_test.datasets["y_graph"].shape[0] == 2


class TestStateDict:
    def test_state_dict_includes_scaler_when_scaling(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_graph=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        sd = dm.state_dict()
        assert "y_scaler" in sd

    def test_state_dict_no_scaler_when_not_scaling(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_graph=False,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        sd = dm.state_dict()
        assert "y_scaler" not in sd


class TestNodeScaling:
    def test_featurize_with_scale_y_node_standardises(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        # Collect all y_node values after scaling
        all_yn = torch.cat([ds[i]["graph"].y_node for i in range(len(small_mol_list))])
        # Mean should be approximately 0 after standard scaling
        assert abs(all_yn.mean().item()) < 0.5

    def test_featurize_without_scale_y_node_unchanged(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_node=False,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        # Values should match the raw input
        for i in range(len(small_mol_list)):
            yn_out = ds[i]["graph"].y_node.numpy()
            np.testing.assert_allclose(yn_out, y_node_small[i], rtol=1e-5)

    def test_fit_y_node_then_transform_on_test(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        # Fit on training data
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        # Transform test data without re-fitting
        ds_test = dm.featurize(
            small_mol_list[:2],
            y_graph_small[:2],
            y_node_small[:2],
            is_training=False,
            n_jobs=1,
        )
        assert ds_test[0]["graph"].y_node.shape[1] == 2

    def test_fit_and_transform_methods(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.generate_features(small_mol_list, y_graph_small, y_node_small, n_jobs=1)
        dm.fit(ds)
        dm.transform(ds)
        all_yn = torch.cat([ds[i]["graph"].y_node for i in range(len(small_mol_list))])
        assert abs(all_yn.mean().item()) < 0.5


class TestNodeScalerStateDict:
    def test_state_dict_includes_node_scaler_when_enabled(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        sd = dm.state_dict()
        assert "y_node_scaler" in sd

    def test_state_dict_excludes_node_scaler_when_disabled(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_node=False,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        sd = dm.state_dict()
        assert "y_node_scaler" not in sd

    def test_load_state_dict_roundtrip_node_scaler(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        sd = dm.state_dict()

        dm2 = GraphPretrainingDataModule.dummy()
        dm2.load_state_dict(sd)
        assert hasattr(dm2._y_node_scaler, "n_features_in_")
        assert dm2._y_node_scaler.n_features_in_ == 2
