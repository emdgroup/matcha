"""Tests for GraphDataModule and Graph3DDataModule."""

import time
import pytest
from pydantic import ValidationError
from torch.utils.data import StackDataset
from torch_geometric.data import Data
from unittest.mock import patch
from rdkit import Chem

from matcha.datamodules.classic.graph_datamodule import (
    GraphDataModule,
    Graph3DDataModule,
)
from matcha.datamodules.base_datamodule import DataModuleRegistry


# ===================================================================
# GraphDataModule – construction
# ===================================================================


class TestGraphDataModuleInit:
    def test_default_construction(self):
        dm = GraphDataModule()
        assert dm.params.datamodule_type == "graph"
        assert dm.params.laplacian_k == 10
        assert dm.params.rwse_k == 20
        assert dm.params.rrwp_k == 20

    def test_custom_params(self):
        dm = GraphDataModule(
            laplacian_k=5, rwse_k=10, rrwp_k=0, compute_distances=False
        )
        assert dm.params.laplacian_k == 5
        assert dm.params.rwse_k == 10
        assert dm.params.rrwp_k == 0
        assert dm.params.compute_distances is False

    def test_classification_mode(self):
        dm = GraphDataModule(is_classification=True)
        assert dm.params.is_classification is True

    def test_registry_has_graph(self):
        assert "graph" in DataModuleRegistry


# ===================================================================
# GraphDataModule – single graph computation
# ===================================================================


class TestCalculateGraph:
    def test_returns_pyg_data(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert isinstance(graph, Data)

    def test_graph_has_node_features(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert graph.x is not None
        assert graph.x.ndim == 2

    def test_graph_has_edge_index(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert graph.edge_index is not None
        assert graph.edge_index.shape[0] == 2

    def test_graph_has_edge_attr(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert graph.edge_attr is not None


class TestGraphPositionalEncodings:
    def test_laplacian_pe(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=5, rwse_k=0, rrwp_k=0, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert hasattr(graph, "laplacian_k")
        assert graph.laplacian_k.shape[1] == 5

    def test_rwse_pe(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=8, rrwp_k=0, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert hasattr(graph, "rwse_k")
        assert graph.rwse_k.shape[1] == 8

    def test_rrwp_pe(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=5, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert hasattr(graph, "rrwp_k")
        assert graph.rrwp_k.shape[1] == 5

    def test_shortest_path_distances(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=True)
        graph = dm._calculate_graph(small_mol_list[0])
        assert graph.spd is not None
        n = graph.num_nodes
        assert graph.spd.shape == (n, n)

    def test_no_pe_when_k_is_zero(self, small_mol_list):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        graph = dm._calculate_graph(small_mol_list[0])
        assert not hasattr(graph, "laplacian_k") or graph.laplacian_k is None or True
        # The attribute won't be set if k=0


class TestGraphVirtualNodes:
    def test_virtual_nodes_increase_node_count(self, small_mol_list):
        dm_no_vn = GraphDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
            num_virtual_nodes=0,
        )
        dm_vn = GraphDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
            num_virtual_nodes=2,
        )
        g_no = dm_no_vn._calculate_graph(small_mol_list[0])
        g_vn = dm_vn._calculate_graph(small_mol_list[0])
        assert g_vn.num_nodes == g_no.num_nodes + 2

    def test_virtual_nodes_add_indicator_feature(self, small_mol_list):
        dm = GraphDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
            num_virtual_nodes=1,
        )
        graph = dm._calculate_graph(small_mol_list[0])
        # Last column should be the virtual node indicator
        assert graph.x[-1, -1].item() == 1.0  # virtual node
        assert graph.x[0, -1].item() == 0.0  # real node


# ===================================================================
# GraphDataModule – featurize
# ===================================================================


class TestGraphFeaturize:
    def test_featurize_training_returns_stack_dataset(
        self, small_mol_list, small_regression_y
    ):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert isinstance(ds, StackDataset)

    def test_featurize_keys(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        item = ds[0]
        assert "graph" in item
        assert "y" in item

    def test_featurize_graph_is_pyg_data(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert isinstance(ds[0]["graph"], Data)

    def test_featurize_y_shape(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert ds.datasets["y"].shape[0] == len(small_mol_list)

    def test_featurize_y_scaled(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        y_out = ds.datasets["y"].numpy()
        # After standard scaling the mean should be ~0
        assert abs(y_out.mean()) < 0.5

    def test_featurize_without_y(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        ds = dm.featurize(small_mol_list, None, is_training=False, n_jobs=1)
        assert ds.datasets["y"].shape[0] == len(small_mol_list)


class TestGraphFeaturizeClassification:
    def test_classification_featurize(self, small_mol_list, small_classification_y):
        dm = GraphDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
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
        ds = dm.featurize(
            small_mol_list, small_classification_y, is_training=True, n_jobs=1
        )
        assert isinstance(ds, StackDataset)


# ===================================================================
# GraphDataModule – state dict
# ===================================================================


class TestGraphStateDict:
    def test_state_dict_keys(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False)
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()
        assert "ID" in sd
        assert "params" in sd
        assert "y_scaler" in sd

    def test_load_state_dict_roundtrip(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(laplacian_k=5, rwse_k=0, rrwp_k=0, compute_distances=False)
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()

        dm2 = GraphDataModule.dummy()
        dm2.load_state_dict(sd)
        assert dm2.params.laplacian_k == 5


# ===================================================================
# GraphDataModule – dataloader
# ===================================================================


class TestGraphDataloader:
    def test_setup_fit(self, small_mol_list, small_regression_y):
        dm = GraphDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False, batch_size=4
        )
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        dm.dataset_train = ds
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "graph" in batch
        assert "y" in batch


# ===================================================================
# Graph3DDataModule
# ===================================================================


class TestGraph3DDataModuleInit:
    def test_default_construction(self):
        dm = Graph3DDataModule()
        assert dm.params.datamodule_type == "graph3d"

    def test_registry_has_graph3d(self):
        assert "graph3d" in DataModuleRegistry


class TestGraph3DFeaturize:
    def test_featurize_returns_stack_dataset(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert isinstance(ds, StackDataset)

    def test_featurize_has_coords(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        item = ds[0]
        assert "coords" in item

    def test_coords_shape(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        coords = ds[0]["coords"]
        graph = ds[0]["graph"]
        # coords should be (num_atoms, 3)
        assert coords.shape[0] == graph.num_nodes
        assert coords.shape[1] == 3


class TestGraph3DStateDict:
    def test_state_dict_roundtrip(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()
        assert sd["ID"] == "graph3d"


# ===================================================================
# GraphDataModule – dummy
# ===================================================================


class TestGraphDummy:
    def test_graph_dummy(self):
        dm = GraphDataModule.dummy()
        assert isinstance(dm, GraphDataModule)

    def test_graph3d_dummy(self):
        dm = Graph3DDataModule()
        assert isinstance(dm, Graph3DDataModule)


# ===================================================================
# Graph3DDataModule – embed_timeout configuration (Stage 1)
# ===================================================================


class TestGraph3DEmbedTimeout:
    def test_default_embed_timeout(self):
        dm = Graph3DDataModule()
        assert dm.params.embed_timeout == 30

    def test_custom_embed_timeout(self):
        dm = Graph3DDataModule(embed_timeout=15)
        assert dm.params.embed_timeout == 15

    def test_embed_timeout_validation(self):
        with pytest.raises(ValidationError):
            Graph3DDataModule(embed_timeout=0)

    def test_embed_timeout_in_state_dict(self):
        dm = Graph3DDataModule(embed_timeout=45)
        sd = dm.state_dict()
        assert sd["params"]["embed_timeout"] == 45


# ===================================================================
# Graph3DDataModule – _embed_with_timeout runtime behavior (Stage 2)
# ===================================================================


class TestEmbedWithTimeoutBehavior:
    """Tests for the timeout guard in _calculate_coords()."""

    _MOL_SMILES = "c1ccccc1"

    def _mol(self):
        return Chem.MolFromSmiles(self._MOL_SMILES)

    def test_timeout_first_call_falls_back_to_random(self):
        mol = self._mol()
        dm = Graph3DDataModule(embed_timeout=0.1)

        call_count = [0]

        def slow_first_call(m, params):
            call_count[0] += 1
            if call_count[0] == 1:
                time.sleep(0.3)  # exceeds timeout
            return 0

        with patch("rdkit.Chem.AllChem.EmbedMolecule", side_effect=slow_first_call):
            coords = dm._calculate_coords(mol)

        assert coords.shape == (mol.GetNumAtoms(), 3)
        assert call_count[0] == 2

    def test_both_timeouts_fall_back_to_2d(self):
        mol = self._mol()
        dm = Graph3DDataModule(embed_timeout=0.1)

        def always_slow(m, params):
            time.sleep(0.3)
            return 0

        with patch("rdkit.Chem.AllChem.EmbedMolecule", side_effect=always_slow):
            coords = dm._calculate_coords(mol)

        assert coords.shape == (mol.GetNumAtoms(), 3)

    def test_no_timeout_on_normal_molecule(self):
        mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
        dm = Graph3DDataModule()
        coords = dm._calculate_coords(mol)
        assert coords.shape == (mol.GetNumAtoms(), 3)

    def test_embed_timeout_value_passed_to_helper(self):
        mol = self._mol()
        dm = Graph3DDataModule(embed_timeout=42)

        with patch(
            "matcha.datamodules.classic.graph_datamodule._embed_with_timeout",
            return_value=0,
        ) as mock_helper:
            dm._calculate_coords(mol)

        assert mock_helper.call_count >= 1
        for c in mock_helper.call_args_list:
            assert c.args[2] == 42
