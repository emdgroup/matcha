"""Tests for GraphPretrainingDataModule and OnTheFlyGraphPretrainingDataModule."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch.utils.data import StackDataset
from torch_geometric.data import Data

from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)
from matcha.datamodules.pretraining.on_the_fly_graph_pretraining_datamodule import (
    OnTheFlyGraphPretrainingDataModule,
)
from matcha.datamodules.base_datamodule import DataModuleRegistry


# ===================================================================
# Fixtures
# ===================================================================


def _make_y_node(mol_list: list, num_targets: int = 2) -> list[np.ndarray]:
    """Generate random per-atom labels aligned to canonical SMILES ordering."""
    rng = np.random.default_rng(42)
    y_node = []
    for mol in mol_list:
        # GraphDataModule re-parses via canonical SMILES, so count atoms on
        # the canonical form to stay consistent with validation.
        canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        n_atoms = canonical.GetNumAtoms()
        y_node.append(rng.standard_normal((n_atoms, num_targets)).astype(np.float32))
    return y_node


@pytest.fixture()
def y_node_small(small_mol_list) -> list[np.ndarray]:
    """Per-atom labels for *small_mol_list* (5 mols, 2 targets)."""
    return _make_y_node(small_mol_list, num_targets=2)


@pytest.fixture()
def y_node(mol_list) -> list[np.ndarray]:
    """Per-atom labels for *mol_list* (30 mols, 2 targets)."""
    return _make_y_node(mol_list, num_targets=2)


@pytest.fixture()
def y_graph_small(small_regression_y) -> np.ndarray:
    """Molecule-level labels for *small_mol_list* (aliased from regression_y)."""
    return small_regression_y


@pytest.fixture()
def y_graph(regression_y) -> np.ndarray:
    """Molecule-level labels for *mol_list* (aliased from regression_y)."""
    return regression_y


# ===================================================================
# Construction
# ===================================================================


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


# ===================================================================
# Validation
# ===================================================================


class TestNodeLabelValidation:
    def test_mismatched_length_raises(self, small_mol_list):
        dm = GraphPretrainingDataModule()
        y_node_wrong = _make_y_node(small_mol_list[:3], num_targets=2)
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


# ===================================================================
# Graph construction with node labels
# ===================================================================


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


# ===================================================================
# generate_features
# ===================================================================


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


# ===================================================================
# featurize
# ===================================================================


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


# ===================================================================
# Collation
# ===================================================================


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


# ===================================================================
# Dataloader integration
# ===================================================================


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


# ===================================================================
# State dict
# ===================================================================


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


# ===================================================================
# Dummy
# ===================================================================


class TestDummy:
    def test_dummy_creation(self):
        dm = GraphPretrainingDataModule.dummy()
        assert isinstance(dm, GraphPretrainingDataModule)
        assert dm.params.datamodule_type == "graph_pretraining"


# ===================================================================
# export_to_classic
# ===================================================================

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


# ===================================================================
# OnTheFlyGraphPretrainingDataModule
# ===================================================================


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


# ===================================================================
# Node scaling (scale_y_node)
# ===================================================================


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


# ===================================================================
# Node scaler state_dict
# ===================================================================


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


# ===================================================================
# On-the-fly collation with scaling
# ===================================================================


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
