"""Tests for Graph3DDataModule construction and data handling."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from rdkit import Chem
from torch.utils.data import StackDataset

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.classic.graph_datamodule import Graph3DDataModule


def _make_user_coords(mol_list) -> list[np.ndarray]:
    """Deterministic 3D coords in the *input* atom order.

    Row i is ``(m_idx, atom_idx, atom_idx)`` — every input-atom index maps to
    a unique 3-vector so we can detect canonical reordering by inspecting
    ``graph.pos``.
    """
    coords = []
    for m_idx, mol in enumerate(mol_list):
        n = mol.GetNumAtoms()
        rows = np.stack(
            [
                np.full(n, float(m_idx)),
                np.arange(n, dtype=np.float32),
                np.arange(n, dtype=np.float32),
            ],
            axis=1,
        )
        coords.append(rows.astype(np.float32))
    return coords


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
            small_mol_list,
            small_regression_y,
            is_training=True,
            n_jobs=1,
            coords=_make_user_coords(small_mol_list),
        )
        assert isinstance(ds, StackDataset)

    def test_featurize_attaches_pos(self, small_mol_list, small_regression_y):
        """Coords ride on ``graph.pos`` — a separate ``coords`` batch key is no
        longer emitted, since PyG :class:`Batch` auto-concatenates ``pos``
        alongside ``x``.
        """
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            n_jobs=1,
            coords=_make_user_coords(small_mol_list),
        )
        item = ds[0]
        assert "coords" not in item
        assert item["graph"].pos is not None

    def test_pos_shape(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            n_jobs=1,
            coords=_make_user_coords(small_mol_list),
        )
        graph = ds[0]["graph"]
        # graph.pos should be (num_atoms, 3), matching graph.x rows
        assert graph.pos.shape[0] == graph.num_nodes
        assert graph.pos.shape[1] == 3

    def test_collate_produces_batched_pos(self, small_mol_list, small_regression_y):
        """After collation, ``bg.pos`` matches ``bg.x`` on the node dim and
        the batch dict does not carry a separate ``coords`` tensor.
        """
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            n_jobs=1,
            coords=_make_user_coords(small_mol_list),
        )
        batch = dm.collate_fn([ds[i] for i in range(len(ds))])
        assert "coords" not in batch
        bg = batch["graph"]
        assert bg.pos is not None
        assert bg.pos.shape == (bg.x.shape[0], 3)


class TestGraph3DStateDict:
    def test_state_dict_roundtrip(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            n_jobs=1,
            coords=_make_user_coords(small_mol_list),
        )
        sd = dm.state_dict()
        assert sd["ID"] == "graph3d"


class TestGraph3DDummy:
    def test_graph3d_dummy(self):
        dm = Graph3DDataModule.dummy()
        assert isinstance(dm, Graph3DDataModule)
        assert dm.params.datamodule_type == "graph3d"
        assert dm.params.embed_timeout == 120.0


class TestGraph3DEmbedTimeout:
    def test_default_embed_timeout(self):
        dm = Graph3DDataModule()
        assert dm.params.embed_timeout == 120

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


class TestGraph3DUserCoordsRouting:
    def test_user_supplied_coords_path_does_not_invoke_embed_and_minimize(
        self, small_mol_list, small_regression_y
    ):
        """Regression: ``featurize(coords=...)`` must skip ETKDG entirely,
        so ``_embed_and_minimize`` is never called.
        """
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = _make_user_coords(small_mol_list)

        with patch(
            "matcha.datamodules.classic.graph_datamodule._embed_and_minimize",
        ) as mock_helper:
            dm.featurize(
                small_mol_list,
                small_regression_y,
                is_training=True,
                n_jobs=1,
                coords=coords,
            )

        assert mock_helper.call_count == 0


class TestGraph3DDataModuleUserCoords:
    """Threading optional ``coords`` through ``featurize`` /
    ``generate_features`` — see issue #72.
    """

    def test_featurize_with_coords_returns_stack_dataset(
        self, small_mol_list, small_regression_y
    ):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = _make_user_coords(small_mol_list)
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            n_jobs=1,
            coords=coords,
        )
        assert isinstance(ds, StackDataset)

    def test_featurize_with_coords_pos_shape(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = _make_user_coords(small_mol_list)
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            n_jobs=1,
            coords=coords,
        )
        graph = ds[0]["graph"]
        assert graph.pos.shape[0] == graph.num_nodes
        assert graph.pos.shape[1] == 3

    def test_featurize_with_coords_skips_etkdg(
        self, small_mol_list, small_regression_y
    ):
        """When coords are supplied, ``_calculate_coords`` (ETKDG) must
        never be called. Monkeypatch it to raise so we catch any leak.
        """
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = _make_user_coords(small_mol_list)

        def _boom(_self, _mol):
            raise AssertionError("_calculate_coords should not be called")

        with patch.object(Graph3DDataModule, "_calculate_coords", new=_boom):
            ds = dm.featurize(
                small_mol_list,
                small_regression_y,
                is_training=True,
                n_jobs=1,
                coords=coords,
            )
        assert isinstance(ds, StackDataset)

    def test_user_coords_honoured_after_canonical_reorder(self):
        """User coords must be reordered to the canonical atom ordering
        before being attached to ``graph.pos``. Uses a SMILES whose canonical
        form permutes atoms so the reorder is observable.
        """
        smi = "OCC(=O)N"  # glycolamide — non-trivial canonicalisation
        mol = Chem.MolFromSmiles(smi)
        n = mol.GetNumAtoms()
        # row i encodes the input atom index i in every column
        coords = np.tile(np.arange(n, dtype=np.float32).reshape(-1, 1), (1, 3))
        y = np.zeros((1, 1), dtype=np.float32)

        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize([mol], y, is_training=True, n_jobs=1, coords=[coords])

        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        match = mol.GetSubstructMatch(canonical_mol)
        expected = np.array(match, dtype=np.float32)

        pos = ds[0]["graph"].pos
        assert torch.allclose(pos[:n, 0], torch.tensor(expected))
        assert torch.allclose(pos[:n, 1], torch.tensor(expected))
        assert torch.allclose(pos[:n, 2], torch.tensor(expected))

    def test_virtual_node_padding_is_zero(self):
        """With ``num_virtual_nodes > 0``, the trailing rows of
        ``graph.pos`` must be exactly zero (not NaN, not left uninitialised).
        """
        mol = Chem.MolFromSmiles("CCO")
        n = mol.GetNumAtoms()
        coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3) + 1.0
        y = np.zeros((1, 1), dtype=np.float32)

        dm = Graph3DDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
            num_virtual_nodes=2,
        )
        ds = dm.featurize([mol], y, is_training=True, n_jobs=1, coords=[coords])

        pos = ds[0]["graph"].pos
        assert pos.shape[0] == n + 2
        assert pos.shape[1] == 3
        assert torch.all(pos[n:] == 0.0)

    def test_wrong_length_raises(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = _make_user_coords(small_mol_list)[:2]  # length mismatch
        with pytest.raises(ValueError, match="coords length"):
            dm.featurize(
                small_mol_list,
                small_regression_y,
                is_training=True,
                n_jobs=1,
                coords=coords,
            )

    def test_nan_coords_raises(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = _make_user_coords(small_mol_list)
        coords[1][0, 0] = np.nan
        with pytest.raises(ValueError, match=r"coords\[1\] contains non-finite"):
            dm.featurize(
                small_mol_list,
                small_regression_y,
                is_training=True,
                n_jobs=1,
                coords=coords,
            )

    def test_augment_resonance_with_coords_raises(
        self, small_mol_list, small_regression_y
    ):
        dm = Graph3DDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
            augment_resonance=True,
        )
        coords = _make_user_coords(small_mol_list)
        with pytest.raises(ValueError, match="augment_resonance"):
            dm.featurize(
                small_mol_list,
                small_regression_y,
                is_training=True,
                n_jobs=1,
                coords=coords,
            )

    def test_coords_none_uses_etkdg_path(self, small_mol_list, small_regression_y):
        """Regression: ``coords=None`` (the default) must route through
        ``_process_batch`` (ETKDG), not ``_process_batch_with_coords``.
        """
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        with (
            patch.object(
                Graph3DDataModule,
                "_process_batch",
                wraps=dm._process_batch,
            ) as mocked_etkdg,
            patch.object(
                Graph3DDataModule,
                "_process_batch_with_coords",
            ) as mocked_user,
            patch.object(
                dm,
                "_calculate_coords",
                side_effect=lambda mol: torch.zeros((mol.GetNumAtoms(), 3)),
            ),
        ):
            ds = dm.featurize(
                small_mol_list, small_regression_y, is_training=True, n_jobs=1
            )

        assert mocked_etkdg.call_count >= 1
        assert mocked_user.call_count == 0
        # pos shape identical to today's ETKDG path
        assert ds[0]["graph"].pos.shape[1] == 3
        assert ds[0]["graph"].pos.shape[0] == ds[0]["graph"].num_nodes

    def test_generate_features_with_coords_routes_to_user_batch(
        self, small_mol_list, small_regression_y
    ):
        """When coords are supplied, ``_process_batch_with_coords`` is
        used and the ETKDG ``_process_batch`` is not.
        """
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = _make_user_coords(small_mol_list)

        with (
            patch.object(
                Graph3DDataModule,
                "_process_batch",
            ) as mocked_etkdg,
            patch.object(
                Graph3DDataModule,
                "_process_batch_with_coords",
                wraps=dm._process_batch_with_coords,
            ) as mocked_user,
        ):
            dm.generate_features(
                small_mol_list, small_regression_y, None, 1, coords=coords
            )

        assert mocked_etkdg.call_count == 0
        assert mocked_user.call_count >= 1
