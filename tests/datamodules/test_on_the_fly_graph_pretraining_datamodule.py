"""Tests for on-the-fly graph pretraining behavior."""

import numpy as np
import torch
from rdkit import Chem

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)
from matcha.datamodules.pretraining.on_the_fly_graph_pretraining_datamodule import (
    OnTheFlyGraphPretrainingDataModule,
)


class TestOnTheFlyGraphPretrainingInit:
    def test_construction(self):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        assert dm.params.datamodule_type == "graph_pretraining"

    def test_registry_has_on_the_fly_graph_pretraining(self):
        assert "on_the_fly_graph_pretraining" in DataModuleRegistry


class TestOnTheFlySetData:
    def test_set_train_data(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        dm.set_data(
            train_smiles=smiles_list[:10],
            train_y_graph=y_graph[:10],
            train_y_node=y_node[:10],
        )
        assert dm._raw_train is not None
        assert len(dm._raw_train) == 10

    def test_set_train_and_val_data(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        dm.set_data(
            train_smiles=smiles_list[:10],
            train_y_graph=y_graph[:10],
            train_y_node=y_node[:10],
            val_smiles=smiles_list[10:15],
            val_y_graph=y_graph[10:15],
            val_y_node=y_node[10:15],
        )
        assert dm._raw_train is not None
        assert dm._raw_val is not None
        assert len(dm._raw_train) == 10
        assert len(dm._raw_val) == 5

    def test_dataset_getitem(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        dm.set_data(
            train_smiles=smiles_list[:5],
            train_y_graph=y_graph[:5],
            train_y_node=y_node[:5],
        )
        item = dm._raw_train[0]
        assert "smiles" in item
        assert "y_graph" in item
        assert "y_node" in item


class TestOnTheFlyCollate:
    def test_collate_fn_produces_correct_keys(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)

        batch = [
            {"smiles": smiles_list[i], "y_graph": y_graph[i], "y_node": y_node[i]}
            for i in range(3)
        ]
        result = dm.collate_fn(batch)
        assert "graph" in result
        assert "y_node" in result
        assert "y_graph" in result

    def test_collate_fn_shapes(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)

        n = 3
        batch = [
            {"smiles": smiles_list[i], "y_graph": y_graph[i], "y_node": y_node[i]}
            for i in range(n)
        ]
        result = dm.collate_fn(batch)
        assert result["y_graph"].shape[0] == n
        # y_node is concatenated across molecules
        total_atoms = sum(y_node[i].shape[0] for i in range(n))
        assert result["y_node"].shape == torch.Size([total_atoms, 2])


class TestOnTheFlyStateDict:
    def test_state_dict_keys(self):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base, num_workers=2)
        sd = dm.state_dict()
        assert sd["ID"] == "on_the_fly_graph_pretraining"
        assert "base_state_dict" in sd
        assert sd["num_workers"] == 2

    def test_load_state_dict(self):
        base = GraphPretrainingDataModule(
            laplacian_k=5,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base, num_workers=4)
        sd = dm.state_dict()

        base2 = GraphPretrainingDataModule()
        dm2 = OnTheFlyGraphPretrainingDataModule(base=base2)
        dm2.load_state_dict(sd)
        assert dm2.params.laplacian_k == 5
        assert dm2.num_workers == 4


class TestNodeScalerStateDict:
    def test_on_the_fly_state_dict_roundtrip_node_scaler(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        base = GraphPretrainingDataModule(
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        base.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        sd = dm.state_dict()

        base2 = GraphPretrainingDataModule()
        dm2 = OnTheFlyGraphPretrainingDataModule(base=base2)
        dm2.load_state_dict(sd)
        assert hasattr(dm2.base._y_node_scaler, "n_features_in_")
        assert dm2.base._y_node_scaler.n_features_in_ == 2


class TestOnTheFlyScaling:
    def test_collate_with_scale_y_graph(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            scale_y_graph=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        # Fit the scaler on a subset using featurize
        mols = [Chem.MolFromSmiles(s) for s in smiles_list[:10]]
        base.featurize(mols, y_graph[:10], y_node[:10], is_training=True, n_jobs=1)

        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        batch = [
            {"smiles": smiles_list[i], "y_graph": y_graph[i], "y_node": y_node[i]}
            for i in range(5)
        ]
        result = dm.collate_fn(batch)
        # Scaled graph targets should have smaller magnitude than raw
        assert result["y_graph"].shape[0] == 5

    def test_collate_with_scale_y_node(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        mols = [Chem.MolFromSmiles(s) for s in smiles_list[:10]]
        base.featurize(mols, y_graph[:10], y_node[:10], is_training=True, n_jobs=1)

        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        batch = [
            {"smiles": smiles_list[i], "y_graph": y_graph[i], "y_node": y_node[i]}
            for i in range(5)
        ]
        result = dm.collate_fn(batch)
        assert result["y_node"].shape[1] == 2

    def test_collate_without_scaling_unchanged(self, smiles_list, y_graph, y_node):
        base = GraphPretrainingDataModule(
            scale_y_graph=False,
            scale_y_node=False,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        batch = [
            {"smiles": smiles_list[i], "y_graph": y_graph[i], "y_node": y_node[i]}
            for i in range(3)
        ]
        result = dm.collate_fn(batch)
        # y_graph values should match the raw input
        expected_y_graph = np.array([y_graph[i] for i in range(3)], dtype=np.float32)
        np.testing.assert_allclose(
            result["y_graph"].numpy(), expected_y_graph, rtol=1e-5
        )
