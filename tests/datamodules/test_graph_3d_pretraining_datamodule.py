"""Tests for Graph3DPretrainingDataModule."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch.utils.data import StackDataset
from torch_geometric.data import Data

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.classic.graph_datamodule import Graph3DDataModule
from matcha.datamodules.pretraining.graph_3d_pretraining_datamodule import (
    Graph3DPretrainingDataModule,
)


# ===================================================================
# Fixtures
# ===================================================================


def _canonical_num_atoms(mol) -> int:
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    return canonical.GetNumAtoms()


def _make_y_node(mol_list: list, num_targets: int = 2) -> list[np.ndarray]:
    """Per-atom labels in canonical atom order."""
    rng = np.random.default_rng(42)
    y_node = []
    for mol in mol_list:
        n_atoms = _canonical_num_atoms(mol)
        y_node.append(rng.standard_normal((n_atoms, num_targets)).astype(np.float32))
    return y_node


def _make_coords(mol_list: list) -> list[np.ndarray]:
    """Deterministic 3D coordinates in the *input* mol atom order.

    Each atom gets a distinct signature so that a correct canonical reorder
    is observable in the resulting ``graph.pos``.
    """
    coords = []
    for m_idx, mol in enumerate(mol_list):
        n_atoms = mol.GetNumAtoms()
        # Row i is (m_idx, atom_idx, atom_idx) — every input-atom index maps
        # to a unique 3-vector so we can detect reordering.
        rows = np.stack(
            [
                np.full(n_atoms, float(m_idx)),
                np.arange(n_atoms, dtype=np.float32),
                np.arange(n_atoms, dtype=np.float32),
            ],
            axis=1,
        )
        coords.append(rows.astype(np.float32))
    return coords


@pytest.fixture()
def y_node_small(small_mol_list) -> list[np.ndarray]:
    return _make_y_node(small_mol_list, num_targets=2)


@pytest.fixture()
def y_graph_small(small_regression_y) -> np.ndarray:
    return small_regression_y


@pytest.fixture()
def coords_small(small_mol_list) -> list[np.ndarray]:
    return _make_coords(small_mol_list)


# ===================================================================
# Construction
# ===================================================================


class TestGraph3DPretrainingInit:
    def test_default_construction(self):
        dm = Graph3DPretrainingDataModule()
        assert dm.params.datamodule_type == "graph3d_pretraining"
        assert dm.params.scale_y_graph is False
        assert dm.params.scale_y_node is False

    def test_custom_params(self):
        dm = Graph3DPretrainingDataModule(
            scale_y_graph=True,
            laplacian_k=5,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        assert dm.params.scale_y_graph is True
        assert dm.params.laplacian_k == 5

    def test_classification_forced_off(self):
        dm = Graph3DPretrainingDataModule()
        assert dm.params.is_classification is False

    def test_registry_has_graph3d_pretraining(self):
        assert "graph3d_pretraining" in DataModuleRegistry

    def test_registry_maps_to_class(self):
        assert DataModuleRegistry["graph3d_pretraining"] is Graph3DPretrainingDataModule


# ===================================================================
# Coord validation
# ===================================================================


class TestCoordValidation:
    def test_mismatched_length_raises(self, small_mol_list, coords_small):
        dm = Graph3DPretrainingDataModule()
        with pytest.raises(ValueError, match="coords length"):
            dm._validate_coords(small_mol_list, coords_small[:2])

    def test_wrong_atom_count_raises(self, small_mol_list):
        dm = Graph3DPretrainingDataModule()
        bad = []
        for mol in small_mol_list:
            n = _canonical_num_atoms(mol)
            bad.append(np.zeros((n + 1, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="rows but molecule"):
            dm._validate_coords(small_mol_list, bad)

    def test_wrong_last_dim_raises(self, small_mol_list):
        dm = Graph3DPretrainingDataModule()
        bad = []
        for mol in small_mol_list:
            n = _canonical_num_atoms(mol)
            bad.append(np.zeros((n, 2), dtype=np.float32))
        with pytest.raises(ValueError, match=r"shape \(num_atoms, 3\)"):
            dm._validate_coords(small_mol_list, bad)

    def test_1d_raises(self, small_mol_list):
        dm = Graph3DPretrainingDataModule()
        bad = [np.zeros(3, dtype=np.float32) for _ in small_mol_list]
        with pytest.raises(ValueError, match=r"shape \(num_atoms, 3\)"):
            dm._validate_coords(small_mol_list, bad)

    def test_valid_coords_pass(self, small_mol_list, coords_small):
        dm = Graph3DPretrainingDataModule()
        validated = dm._validate_coords(small_mol_list, coords_small)
        assert len(validated) == len(small_mol_list)
        for ci, mol in zip(validated, small_mol_list):
            assert ci.shape == (_canonical_num_atoms(mol), 3)
            assert ci.dtype == np.float32


# ===================================================================
# Canonical coord reorder
# ===================================================================


class TestCanonicalCoordReorder:
    def _distinct_coords(self, n_atoms: int) -> np.ndarray:
        """Coords where row i = [i, i+100, i+200] — each row is unique."""
        i = np.arange(n_atoms, dtype=np.float32)
        return np.stack([i, i + 100, i + 200], axis=1)

    def test_reorder_matches_substruct_match(self):
        # Use a non-canonical SMILES that RDKit will re-canonicalise. If the
        # canonical mol's atom order differs from the input mol's, we can
        # observe the reorder by picking distinct coords.
        smi = "OCC(=O)N"  # glycolamide — chosen for a non-trivial canonicalisation
        mol = Chem.MolFromSmiles(smi)
        n = mol.GetNumAtoms()
        coords = self._distinct_coords(n)

        dm = Graph3DPretrainingDataModule()
        reordered = dm._reorder_coords_to_canonical(mol, coords)

        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        match = mol.GetSubstructMatch(canonical_mol)

        # canonical row i must come from input row match[i]
        for i, m in enumerate(match):
            np.testing.assert_allclose(reordered[i], coords[m])

    def test_pos_rows_follow_canonical_order(self):
        smi = "OCC(=O)N"
        mol = Chem.MolFromSmiles(smi)
        n = mol.GetNumAtoms()
        # coords carry the input atom index in every column, so the value
        # is a stable identifier for the input atom
        coords = np.tile(np.arange(n, dtype=np.float32).reshape(-1, 1), (1, 3))
        y_node = np.zeros((n, 1), dtype=np.float32)

        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels_and_pos(mol, y_node, coords)

        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        match = mol.GetSubstructMatch(canonical_mol)
        expected = np.array(match, dtype=np.float32)
        # graph.pos rows should equal the substruct-match indices
        assert torch.allclose(graph.pos[:, 0], torch.tensor(expected))


# ===================================================================
# Graph construction with node labels and pos
# ===================================================================


class TestCalculateGraphWithLabelsAndPos:
    def test_pos_and_y_node_attached(self, small_mol_list, y_node_small, coords_small):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels_and_pos(
            small_mol_list[0], y_node_small[0], coords_small[0]
        )
        assert isinstance(graph, Data)
        assert hasattr(graph, "pos")
        assert hasattr(graph, "y_node")
        assert graph.pos.shape[1] == 3

    def test_shapes_match_num_nodes_without_virtual(
        self, small_mol_list, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            num_virtual_nodes=0,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels_and_pos(
            small_mol_list[0], y_node_small[0], coords_small[0]
        )
        assert graph.pos.shape[0] == graph.num_nodes
        assert graph.y_node.shape[0] == graph.num_nodes

    def test_virtual_node_pos_padding_is_zero(
        self, small_mol_list, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            num_virtual_nodes=2,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels_and_pos(
            small_mol_list[0], y_node_small[0], coords_small[0]
        )
        assert graph.pos.shape[0] == graph.num_nodes
        # Virtual node coords are zero (not NaN — NaN would poison E3GNN's
        # Fourier distance features on real neighbours of a virtual node)
        assert torch.all(graph.pos[-2:] == 0.0)
        # y_node virtual rows remain NaN (parity with the parent's behaviour)
        assert torch.all(torch.isnan(graph.y_node[-2:]))

    def test_dtype_is_float32(self, small_mol_list, y_node_small, coords_small):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        graph = dm._calculate_graph_with_node_labels_and_pos(
            small_mol_list[0], y_node_small[0], coords_small[0]
        )
        assert graph.pos.dtype == torch.float32


# ===================================================================
# generate_features / featurize
# ===================================================================


class TestGenerateFeatures:
    def test_missing_coords_raises(self, small_mol_list, y_graph_small, y_node_small):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        with pytest.raises(ValueError, match="coords must be provided"):
            dm.generate_features(
                small_mol_list, y_graph_small, y_node_small, coords=None, n_jobs=1
            )

    def test_missing_y_node_raises(self, small_mol_list, y_graph_small, coords_small):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        with pytest.raises(ValueError, match="y_node must be provided"):
            dm.generate_features(
                small_mol_list, y_graph_small, None, coords=coords_small, n_jobs=1
            )

    def test_returns_stack_dataset(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.generate_features(
            small_mol_list, y_graph_small, y_node_small, coords_small, n_jobs=1
        )
        assert isinstance(ds, StackDataset)

    def test_graph_has_pos_and_y_node(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.generate_features(
            small_mol_list, y_graph_small, y_node_small, coords_small, n_jobs=1
        )
        graph = ds[0]["graph"]
        assert hasattr(graph, "pos")
        assert hasattr(graph, "y_node")


class TestFeaturize:
    def test_returns_stack_dataset(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.featurize(
            small_mol_list,
            y_graph_small,
            y_node_small,
            coords_small,
            is_training=True,
            n_jobs=1,
        )
        assert isinstance(ds, StackDataset)

    def test_featurize_with_positional_encodings(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=5,
            rwse_k=8,
            rrwp_k=0,
            compute_distances=False,
        )
        ds = dm.featurize(
            small_mol_list,
            y_graph_small,
            y_node_small,
            coords_small,
            is_training=True,
            n_jobs=1,
        )
        graph = ds[0]["graph"]
        assert graph.laplacian_k.shape[1] == 5
        assert graph.rwse_k.shape[1] == 8
        assert graph.pos.shape[1] == 3


# ===================================================================
# Collation
# ===================================================================


class TestCollation:
    def _make_batch(self, dm, mol_list, y_graph, y_node, coords):
        ds = dm.generate_features(mol_list, y_graph, y_node, coords, n_jobs=1)
        return [ds[i] for i in range(len(mol_list))]

    def test_collate_fn_keys(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(
            dm, small_mol_list, y_graph_small, y_node_small, coords_small
        )
        batch = dm.collate_fn(batch_list)
        assert "graph" in batch
        assert "y_node" in batch
        assert "y_graph" in batch

    def test_batched_pos_shape(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            num_virtual_nodes=0,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(
            dm, small_mol_list, y_graph_small, y_node_small, coords_small
        )
        batch = dm.collate_fn(batch_list)
        total_atoms = sum(_canonical_num_atoms(m) for m in small_mol_list)
        assert batch["graph"].pos.shape == torch.Size([total_atoms, 3])

    def test_batched_pos_shape_with_virtual_nodes(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        n_virtual = 2
        dm = Graph3DPretrainingDataModule(
            num_virtual_nodes=n_virtual,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        batch_list = self._make_batch(
            dm, small_mol_list, y_graph_small, y_node_small, coords_small
        )
        batch = dm.collate_fn(batch_list)
        expected = sum(_canonical_num_atoms(m) + n_virtual for m in small_mol_list)
        assert batch["graph"].pos.shape == torch.Size([expected, 3])


# ===================================================================
# State dict
# ===================================================================


class TestStateDict:
    def test_state_dict_keys(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list,
            y_graph_small,
            y_node_small,
            coords_small,
            is_training=True,
            n_jobs=1,
        )
        sd = dm.state_dict()
        assert sd["ID"] == "graph3d_pretraining"
        assert "params" in sd

    def test_state_dict_scaler_when_enabled(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            scale_y_graph=True,
            scale_y_node=True,
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list,
            y_graph_small,
            y_node_small,
            coords_small,
            is_training=True,
            n_jobs=1,
        )
        sd = dm.state_dict()
        assert "y_scaler" in sd
        assert "y_node_scaler" in sd

    def test_load_state_dict_roundtrip(
        self, small_mol_list, y_graph_small, y_node_small, coords_small
    ):
        dm = Graph3DPretrainingDataModule(
            laplacian_k=5,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm.featurize(
            small_mol_list,
            y_graph_small,
            y_node_small,
            coords_small,
            is_training=True,
            n_jobs=1,
        )
        sd = dm.state_dict()

        dm2 = Graph3DPretrainingDataModule.dummy()
        dm2.load_state_dict(sd)
        assert dm2.params.datamodule_type == "graph3d_pretraining"
        assert dm2.params.laplacian_k == 5


# ===================================================================
# Dummy
# ===================================================================


class TestDummy:
    def test_dummy_creation(self):
        dm = Graph3DPretrainingDataModule.dummy()
        assert isinstance(dm, Graph3DPretrainingDataModule)
        assert dm.params.datamodule_type == "graph3d_pretraining"


# ===================================================================
# export_to_classic
# ===================================================================


class TestExportToClassic:
    def test_returns_graph3d_datamodule(self):
        dm = Graph3DPretrainingDataModule()
        classic = dm.export_to_classic()
        assert isinstance(classic, Graph3DDataModule)
        assert not isinstance(classic, Graph3DPretrainingDataModule)

    def test_params_preserved(self):
        dm = Graph3DPretrainingDataModule(
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
        dm = Graph3DPretrainingDataModule()
        classic = dm.export_to_classic()
        assert classic.params.datamodule_type == "graph3d"

    def test_no_pretraining_specific_attrs(self):
        dm = Graph3DPretrainingDataModule()
        classic = dm.export_to_classic()
        assert not hasattr(classic.params, "scale_y_graph")
