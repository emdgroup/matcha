"""Tests for OnTheFlyGraphPretrainingDataModule coords passthrough.

Covers Stage 2 of issue #29: the wrapper ferries per-molecule coords into
``base.generate_features(..., coords=...)`` when the base is a 3D pretraining
datamodule, and the resulting batch carries ``pos`` aligned with the canonical
atom order.
"""

import numpy as np
import pytest
import torch
from rdkit import Chem

from matcha.datamodules.pretraining.graph_3d_pretraining_datamodule import (
    Graph3DPretrainingDataModule,
)
from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)
from matcha.datamodules.pretraining.on_the_fly_graph_pretraining_datamodule import (
    OnTheFlyGraphPretrainingDataModule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _canonical_num_atoms(mol) -> int:
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    return canonical.GetNumAtoms()


def _make_y_node(mol_list, num_targets: int = 2) -> list[np.ndarray]:
    """Per-atom labels in canonical atom order."""
    rng = np.random.default_rng(0)
    out = []
    for mol in mol_list:
        n_atoms = _canonical_num_atoms(mol)
        out.append(rng.standard_normal((n_atoms, num_targets)).astype(np.float32))
    return out


def _make_coords(mol_list) -> list[np.ndarray]:
    """Distinct per-atom 3-vectors in *input* mol atom order.

    Row i encodes ``(mol_idx, atom_idx, atom_idx)`` so any canonical reorder
    is directly observable in ``graph.pos``.
    """
    coords = []
    for m_idx, mol in enumerate(mol_list):
        n_atoms = mol.GetNumAtoms()
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
def coords_small(small_mol_list) -> list[np.ndarray]:
    return _make_coords(small_mol_list)


@pytest.fixture()
def y_node_small(small_mol_list) -> list[np.ndarray]:
    return _make_y_node(small_mol_list, num_targets=2)


@pytest.fixture()
def smiles_small(small_mol_list) -> list[str]:
    return [Chem.MolToSmiles(m) for m in small_mol_list]


# ---------------------------------------------------------------------------
# 3D base — coords flow through
# ---------------------------------------------------------------------------


class TestCollateForwardsCoordsToGraph3DBase:
    def test_batch_carries_pos_with_expected_shape(
        self,
        smiles_small,
        small_regression_y,
        y_node_small,
        coords_small,
        small_mol_list,
    ):
        base = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        dm.set_data(
            train_smiles=smiles_small,
            train_y_graph=small_regression_y,
            train_y_node=y_node_small,
            train_coords=coords_small,
        )

        batch = [dm._raw_train[i] for i in range(len(smiles_small))]
        # coords should ride on each per-item dict when set_data received them
        assert all("coords" in item for item in batch)

        out = dm.collate_fn(batch)

        assert "graph" in out
        graph = out["graph"]
        assert hasattr(graph, "pos"), "3D base must produce batch.pos"

        total_atoms = sum(_canonical_num_atoms(m) for m in small_mol_list)
        assert graph.pos.shape == torch.Size([total_atoms, 3])
        assert graph.pos.dtype == torch.float32

    def test_pos_matches_canonical_reorder_per_molecule(
        self,
        smiles_small,
        small_regression_y,
        y_node_small,
        coords_small,
        small_mol_list,
    ):
        """graph.pos rows must match user coords after canonical reordering."""
        base = Graph3DPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        dm.set_data(
            train_smiles=smiles_small,
            train_y_graph=small_regression_y,
            train_y_node=y_node_small,
            train_coords=coords_small,
        )

        # Preserve original ordering: no shuffle, no filtering.
        batch = [dm._raw_train[i] for i in range(len(smiles_small))]
        out = dm.collate_fn(batch)

        # Rebuild the expected canonical-reordered coords using the same
        # helper the base uses internally, so this test asserts alignment
        # end-to-end without hard-coding RDKit's canonical order.
        expected_rows = []
        for mol, ci in zip(small_mol_list, coords_small):
            expected_rows.append(base._reorder_coords_to_canonical(mol, ci))
        expected = np.concatenate(expected_rows, axis=0).astype(np.float32)

        assert out["graph"].pos.shape[0] == expected.shape[0]
        np.testing.assert_allclose(
            out["graph"].pos.detach().cpu().numpy(), expected, rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# 2D base — regression: no coords, no pos
# ---------------------------------------------------------------------------


class TestCollateDoesNotAddPosWithout3DBase:
    def test_batch_has_no_pos_when_no_coords_supplied(
        self, smiles_small, small_regression_y, y_node_small
    ):
        base = GraphPretrainingDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
        )
        dm = OnTheFlyGraphPretrainingDataModule(base=base)
        dm.set_data(
            train_smiles=smiles_small,
            train_y_graph=small_regression_y,
            train_y_node=y_node_small,
        )

        batch = [dm._raw_train[i] for i in range(len(smiles_small))]
        assert all("coords" not in item for item in batch)

        out = dm.collate_fn(batch)
        # Parent (2D) GraphPretrainingDataModule never attaches pos.
        assert getattr(out["graph"], "pos", None) is None
