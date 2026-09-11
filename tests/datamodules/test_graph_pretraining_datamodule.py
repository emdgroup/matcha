"""Tests for offline graph pretraining behavior."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch.utils.data import StackDataset
from torch_geometric.data import Data

from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)
from matcha.datamodules.base_datamodule import DataModuleRegistry


class TestGraphPretrainingInit:
    def test_default_construction(self):
        dm = GraphPretrainingDataModule()
        assert dm.params.datamodule_type == "graph_pretraining"
        assert dm.params.scale_y_graph is False
        assert dm.params.scale_y_node is False

    def test_custom_params(self):
        dm = GraphPretrainingDataModule(
            scale_y_graph=True,
            laplacian_k=5,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        assert dm.params.scale_y_graph is True
        assert dm.params.laplacian_k == 5

    def test_classification_forced_off(self):
        dm = GraphPretrainingDataModule()
        assert dm.params.is_classification is False

    def test_registry_has_graph_pretraining(self):
        assert "graph_pretraining" in DataModuleRegistry


class TestNodeLabelValidation:
    def test_mismatched_length_raises(self, small_mol_list, y_node_small):
        dm = GraphPretrainingDataModule()
        y_node_wrong = y_node_small[:3]
        with pytest.raises(ValueError, match="y_node length"):
            dm._validate_node_labels(small_mol_list, y_node_wrong)

    def test_wrong_atom_count_raises(self, small_mol_list):
        dm = GraphPretrainingDataModule()
        y_node_bad = []
        for mol in small_mol_list:
            # Add one extra row to make the count wrong
            canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
            n = canonical.GetNumAtoms()
            y_node_bad.append(np.zeros((n + 1, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="rows but molecule"):
            dm._validate_node_labels(small_mol_list, y_node_bad)

    def test_1d_array_auto_reshaped(self, small_mol_list):
        dm = GraphPretrainingDataModule()
        y_node_1d = []
        for mol in small_mol_list:
            canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
            n = canonical.GetNumAtoms()
            y_node_1d.append(np.zeros(n, dtype=np.float32))  # 1D
        validated = dm._validate_node_labels(small_mol_list, y_node_1d)
        for yn in validated:
            assert yn.ndim == 2
            assert yn.shape[1] == 1

    def test_valid_labels_pass(self, small_mol_list, y_node_small):
        dm = GraphPretrainingDataModule()
        validated = dm._validate_node_labels(small_mol_list, y_node_small)
        assert len(validated) == len(small_mol_list)


class TestCalculateGraphWithNodeLabels:
    def test_y_node_attached(self, small_mol_list, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels(small_mol_list[0], y_node_small[0])
        assert isinstance(graph, Data)
        assert hasattr(graph, "y_node")
        assert graph.y_node.shape[1] == 2

    def test_y_node_matches_num_nodes_without_virtual(
        self, small_mol_list, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            num_virtual_nodes=0,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels(small_mol_list[0], y_node_small[0])
        assert graph.y_node.shape[0] == graph.num_nodes

    def test_y_node_padded_for_virtual_nodes(self, small_mol_list, y_node_small):
        dm = GraphPretrainingDataModule(
            num_virtual_nodes=2,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels(small_mol_list[0], y_node_small[0])
        # y_node should include NaN-padded rows for virtual nodes
        assert graph.y_node.shape[0] == graph.num_nodes
        # Virtual node labels should be NaN (masked out in loss)
        assert torch.all(torch.isnan(graph.y_node[-2:]))

    def test_graph_still_has_standard_attributes(self, small_mol_list, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=5,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels(small_mol_list[0], y_node_small[0])
        assert graph.x is not None
        assert graph.edge_index is not None
        assert hasattr(graph, "laplacian_k")


class TestGenerateFeatures:
    def test_returns_stack_dataset(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.generate_features(small_mol_list, y_graph_small, y_node_small, n_jobs=1)
        assert isinstance(ds, StackDataset)

    def test_dataset_keys(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.generate_features(small_mol_list, y_graph_small, y_node_small, n_jobs=1)
        item = ds[0]
        assert "graph" in item
        assert "y_graph" in item

    def test_graph_has_y_node(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.generate_features(small_mol_list, y_graph_small, y_node_small, n_jobs=1)
        graph = ds[0]["graph"]
        assert hasattr(graph, "y_node")
        assert graph.y_node.shape[1] == 2

    def test_y_graph_shape(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.generate_features(small_mol_list, y_graph_small, y_node_small, n_jobs=1)
        assert ds.datasets["y_graph"].shape == torch.Size(
            [len(small_mol_list), y_graph_small.shape[1]]
        )

    def test_missing_y_node_raises(self, small_mol_list, y_graph_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        with pytest.raises(ValueError, match="y_node must be provided"):
            dm.generate_features(small_mol_list, y_graph_small, None, n_jobs=1)


class TestFeaturize:
    def test_featurize_returns_stack_dataset(
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
        assert isinstance(ds, StackDataset)

    def test_featurize_with_positional_encodings(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            laplacian_k=5,
            rwse_k=8,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        graph = ds[0]["graph"]
        assert hasattr(graph, "laplacian_k")
        assert graph.laplacian_k.shape[1] == 5
        assert hasattr(graph, "rwse_k")
        assert graph.rwse_k.shape[1] == 8


class TestCollation:
    def _make_batch(self, dm, mol_list, y_graph, y_node):
        ds = dm.generate_features(mol_list, y_graph, y_node, n_jobs=1)
        return [ds[i] for i in range(len(mol_list))]

    def test_collate_fn_keys(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(dm, small_mol_list, y_graph_small, y_node_small)
        batch = dm.collate_fn(batch_list)
        assert "graph" in batch
        assert "y_node" in batch
        assert "y_graph" in batch

    def test_collate_y_graph_shape(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(dm, small_mol_list, y_graph_small, y_node_small)
        batch = dm.collate_fn(batch_list)
        assert batch["y_graph"].shape == torch.Size(
            [len(small_mol_list), y_graph_small.shape[1]]
        )

    def test_collate_y_node_shape(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(dm, small_mol_list, y_graph_small, y_node_small)
        batch = dm.collate_fn(batch_list)
        # y_node should be concatenated across all molecules
        total_atoms = sum(yn.shape[0] for yn in y_node_small)
        assert batch["y_node"].shape == torch.Size([total_atoms, 2])

    def test_collate_y_node_not_on_graph(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(dm, small_mol_list, y_graph_small, y_node_small)
        batch = dm.collate_fn(batch_list)
        # y_node should have been removed from the batched graph
        assert not hasattr(batch["graph"], "y_node") or batch["graph"].y_node is None

    def test_collate_graph_has_batch_attr(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(dm, small_mol_list, y_graph_small, y_node_small)
        batch = dm.collate_fn(batch_list)
        assert hasattr(batch["graph"], "batch")


class TestDataloader:
    def test_setup_fit_produces_batches(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
            batch_size=4,
        )
        ds = dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        dm.dataset_train = ds
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "graph" in batch
        assert "y_node" in batch
        assert "y_graph" in batch
        assert batch["y_graph"].shape[0] <= 4


class TestStateDict:
    def test_state_dict_keys(self, small_mol_list, y_graph_small, y_node_small):
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list, y_graph_small, y_node_small, is_training=True, n_jobs=1
        )
        sd = dm.state_dict()
        assert sd["ID"] == "graph_pretraining"
        assert "params" in sd

    def test_load_state_dict_roundtrip(
        self, small_mol_list, y_graph_small, y_node_small
    ):
        dm = GraphPretrainingDataModule(
            laplacian_k=5,
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
        assert dm2.params.laplacian_k == 5


class TestDummy:
    def test_dummy_creation(self):
        dm = GraphPretrainingDataModule.dummy()
        assert isinstance(dm, GraphPretrainingDataModule)
        assert dm.params.datamodule_type == "graph_pretraining"


from matcha.datamodules.classic.graph_datamodule import GraphDataModule  # noqa: E402


class TestExportToClassic:
    def test_returns_graph_datamodule(self):
        dm = GraphPretrainingDataModule()
        classic = dm.export_to_classic()
        assert isinstance(classic, GraphDataModule)
        assert not isinstance(classic, GraphPretrainingDataModule)

    def test_params_preserved(self):
        dm = GraphPretrainingDataModule(
            laplacian_k=5,
            rwse_k=8,
            rrwp_k=12,
            elstatic_k=3,
            distmat_k=4,
            compute_distances=False,
            num_virtual_nodes=2,
            init_virtual_nodes=True,
            batch_size=64,
        )
        classic = dm.export_to_classic()
        assert classic.params.laplacian_k == 5
        assert classic.params.rwse_k == 8
        assert classic.params.rrwp_k == 12
        assert classic.params.elstatic_k == 3
        assert classic.params.distmat_k == 4
        assert classic.params.compute_distances is False
        assert classic.params.num_virtual_nodes == 2
        assert classic.params.init_virtual_nodes is True
        assert classic.params.batch_size == 64

    def test_classic_datamodule_type(self):
        dm = GraphPretrainingDataModule()
        classic = dm.export_to_classic()
        assert classic.params.datamodule_type == "graph"

    def test_classic_can_featurize(self, small_mol_list, y_graph_small):
        """Exported GraphDataModule should be able to featurize with labels."""
        dm = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        classic = dm.export_to_classic()
        ds = classic.featurize(
            small_mol_list,
            y=y_graph_small,
            is_training=True,
            n_jobs=1,
        )
        assert "graph" in ds.datasets
        assert "y" in ds.datasets
        assert ds.datasets["y"].shape[0] == len(small_mol_list)

    def test_classification_setting_preserved(self):
        """is_classification is forced False in pretraining, so it should be False in classic too."""
        dm = GraphPretrainingDataModule()
        classic = dm.export_to_classic()
        assert classic.params.is_classification is False

    def test_no_pretraining_specific_attrs(self):
        """Exported instance should not have pretraining-specific params."""
        dm = GraphPretrainingDataModule()
        classic = dm.export_to_classic()
        assert not hasattr(classic.params, "scale_y_graph")
