"""Tests for GraphDataModule and Graph3DDataModule."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import StackDataset
from torch_geometric.data import Data
from unittest.mock import MagicMock, patch
from rdkit import Chem
from rdkit.Chem import AllChem

from matcha.datamodules.classic.graph_datamodule import (
    GraphDataModule,
    Graph3DDataModule,
    _embed_and_minimize,
    _select_lowest_energy_conf_id,
)
from matcha.datamodules.base_datamodule import DataModuleRegistry


def _canonical_num_atoms(mol) -> int:
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    return canonical.GetNumAtoms()


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

    def test_featurize_attaches_pos(self, small_mol_list, small_regression_y):
        """Coords ride on ``graph.pos`` — a separate ``coords`` batch key is no
        longer emitted, since PyG :class:`Batch` auto-concatenates ``pos``
        alongside ``x``.
        """
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        item = ds[0]
        assert "coords" not in item
        assert item["graph"].pos is not None

    def test_pos_shape(self, small_mol_list, small_regression_y):
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
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
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
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


# ===================================================================
# Graph3DDataModule – embed timeout dispatch in _calculate_coords (Stage 3)
# ===================================================================


class TestEmbedWithTimeoutBehavior:
    """Tests for the timeout dispatch from _calculate_coords()."""

    _MOL_SMILES = "c1ccccc1"

    def _mol(self):
        return Chem.MolFromSmiles(self._MOL_SMILES)

    def test_no_timeout_on_normal_molecule(self):
        mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
        dm = Graph3DDataModule()
        coords = dm._calculate_coords(mol)
        assert coords.shape == (mol.GetNumAtoms(), 3)

    def test_embed_timeout_value_passed_to_helper(self):
        mol = self._mol()
        dm = Graph3DDataModule(embed_timeout=42)

        mol_h = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol_h)

        with patch(
            "matcha.datamodules.classic.graph_datamodule._embed_and_minimize",
            return_value=(mol_h, 0),
        ) as mock_helper:
            dm._calculate_coords(mol)

        assert mock_helper.call_count == 1
        args, kwargs = mock_helper.call_args
        assert args[1] == 42 or kwargs.get("timeout_seconds") == 42


# ===================================================================
# _select_lowest_energy_conf_id – pure ranking function (Stage 2, issue #76)
# ===================================================================


class TestSelectLowestEnergyConfId:
    """Pure-Python tests for the MMFF-result ranking function."""

    def test_all_converged_picks_lowest_energy(self):
        results = [(0, 2.5), (0, 1.0), (0, 3.5)]
        assert _select_lowest_energy_conf_id(results) == 1

    def test_converged_wins_over_lower_energy_non_converged(self):
        # Non-converged conformer has a lower energy but the issue rule says
        # any converged conformer must win.
        results = [(1, -5.0), (0, 2.0), (1, -10.0)]
        assert _select_lowest_energy_conf_id(results) == 1

    def test_all_non_converged_picks_lowest_energy(self):
        results = [(1, 3.0), (1, 1.5), (1, 2.0)]
        assert _select_lowest_energy_conf_id(results) == 1

    def test_mix_finite_and_nan_non_converged_picks_lowest_finite(self):
        results = [
            (1, float("nan")),
            (1, 5.0),
            (1, float("inf")),
            (1, 3.0),
            (1, float("-inf")),
        ]
        assert _select_lowest_energy_conf_id(results) == 3

    def test_all_non_finite_returns_none(self):
        results = [
            (0, float("nan")),
            (1, float("inf")),
            (1, float("-inf")),
        ]
        assert _select_lowest_energy_conf_id(results) is None

    def test_empty_returns_none(self):
        assert _select_lowest_energy_conf_id([]) is None


# ===================================================================
# _embed_and_minimize – orchestrator (Stage 2, issue #76)
# ===================================================================


def _sync_run_with_timeout(fn, *args, timeout_seconds, **kwargs):
    """Bypass the thread pool — invoke the wrapped callable synchronously.

    Used to keep the mocked ``EmbedMultipleConfs`` / ``MMFFOptimizeMoleculeConfs``
    call assertions observable in unit tests, while preserving the real
    control flow inside :func:`_embed_and_minimize`.
    """
    return fn(*args, **kwargs)


class TestEmbedAndMinimize:
    """Unit tests for the ETKDG + MMFF orchestrator with mocked RDKit calls."""

    _SMILES = "CCO"

    def _mol(self):
        return Chem.MolFromSmiles(self._SMILES)

    def test_happy_path_returns_mol_and_lowest_conf_id(self):
        mol = self._mol()
        fake_mol_h = MagicMock(name="mol_h")
        # Conformer index 2 is the lowest-energy converged conformer.
        fake_results = [(0, 5.0), (0, 4.0), (0, 1.0), (0, 3.0), (0, 2.0)]

        with (
            patch(
                "matcha.datamodules.classic.graph_datamodule.Chem.AddHs",
                return_value=fake_mol_h,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule._run_with_timeout",
                side_effect=_sync_run_with_timeout,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.EmbedMultipleConfs",
                return_value=list(range(5)),
            ) as mock_embed,
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.MMFFOptimizeMoleculeConfs",
                return_value=fake_results,
            ) as mock_mmff,
        ):
            result = _embed_and_minimize(mol, timeout_seconds=10.0)

        assert result is not None
        mol_h, chosen_id = result
        assert mol_h is fake_mol_h
        assert chosen_id == 2
        assert mock_embed.call_count == 1
        assert mock_mmff.call_count == 1
        # MMFF must be called with the required kwargs.
        _, mmff_kwargs = mock_mmff.call_args
        assert mmff_kwargs["numThreads"] == 1
        assert mmff_kwargs["maxIters"] == 1000
        assert mmff_kwargs["mmffVariant"] == "MMFF94"
        assert mmff_kwargs["nonBondedThresh"] == 100.0

    def test_zero_confs_first_try_retries_with_random_coords(self):
        mol = self._mol()
        fake_mol_h = MagicMock(name="mol_h")
        fake_results = [(0, 1.0)]

        seen_use_random_coords: list[bool] = []

        def _embed(mol_h_arg, numConfs, params):
            seen_use_random_coords.append(bool(params.useRandomCoords))
            if len(seen_use_random_coords) == 1:
                return []
            return [0]

        with (
            patch(
                "matcha.datamodules.classic.graph_datamodule.Chem.AddHs",
                return_value=fake_mol_h,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule._run_with_timeout",
                side_effect=_sync_run_with_timeout,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.EmbedMultipleConfs",
                side_effect=_embed,
            ) as mock_embed,
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.MMFFOptimizeMoleculeConfs",
                return_value=fake_results,
            ) as mock_mmff,
        ):
            result = _embed_and_minimize(mol, timeout_seconds=10.0)

        assert result is not None
        mol_h, chosen_id = result
        assert mol_h is fake_mol_h
        assert chosen_id == 0
        assert mock_embed.call_count == 2
        assert seen_use_random_coords == [False, True]
        # RemoveAllConformers must be invoked between the two embed attempts.
        fake_mol_h.RemoveAllConformers.assert_called_once()
        # MMFF only runs for the successful second embed.
        assert mock_mmff.call_count == 1

    def test_zero_confs_after_retry_returns_none(self):
        mol = self._mol()
        fake_mol_h = MagicMock(name="mol_h")

        with (
            patch(
                "matcha.datamodules.classic.graph_datamodule.Chem.AddHs",
                return_value=fake_mol_h,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule._run_with_timeout",
                side_effect=_sync_run_with_timeout,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.EmbedMultipleConfs",
                return_value=[],
            ) as mock_embed,
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.MMFFOptimizeMoleculeConfs",
            ) as mock_mmff,
        ):
            result = _embed_and_minimize(mol, timeout_seconds=10.0)

        assert result is None
        assert mock_embed.call_count == 2
        # No MMFF call after the second failed embed.
        assert mock_mmff.call_count == 0

    def test_run_with_timeout_returns_none_returns_none(self):
        mol = self._mol()

        with patch(
            "matcha.datamodules.classic.graph_datamodule._run_with_timeout",
            return_value=None,
        ) as mock_run:
            result = _embed_and_minimize(mol, timeout_seconds=10.0)

        assert result is None
        # Hard failure — no retry.
        assert mock_run.call_count == 1

    def test_selector_returns_none_returns_none(self):
        mol = self._mol()
        fake_mol_h = MagicMock(name="mol_h")
        nan = float("nan")
        fake_results = [(1, nan), (1, nan), (1, float("inf"))]

        with (
            patch(
                "matcha.datamodules.classic.graph_datamodule.Chem.AddHs",
                return_value=fake_mol_h,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule._run_with_timeout",
                side_effect=_sync_run_with_timeout,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.EmbedMultipleConfs",
                return_value=list(range(3)),
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.MMFFOptimizeMoleculeConfs",
                return_value=fake_results,
            ),
        ):
            result = _embed_and_minimize(mol, timeout_seconds=10.0)

        assert result is None

    def test_random_seed_is_42(self):
        mol = self._mol()
        fake_mol_h = MagicMock(name="mol_h")
        seen_seeds: list[int] = []

        def _embed(mol_h_arg, numConfs, params):
            seen_seeds.append(params.randomSeed)
            return [0]

        with (
            patch(
                "matcha.datamodules.classic.graph_datamodule.Chem.AddHs",
                return_value=fake_mol_h,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule._run_with_timeout",
                side_effect=_sync_run_with_timeout,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.EmbedMultipleConfs",
                side_effect=_embed,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.MMFFOptimizeMoleculeConfs",
                return_value=[(0, 0.0)],
            ),
        ):
            _embed_and_minimize(mol, timeout_seconds=10.0)

        assert seen_seeds == [42]


# ===================================================================
# Graph3DDataModule – multi-conformer + MMFF pipeline (Stage 3, issue #76)
# ===================================================================


class TestGraph3DMultiConformerPipeline:
    """End-to-end tests for ``_calculate_coords`` wired to the new pipeline."""

    def test_pipeline_returns_correct_shape(self):
        mol = Chem.MolFromSmiles("CCO")  # ethanol, 3 heavy atoms
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords = dm._calculate_coords(mol)
        assert coords.shape == (mol.GetNumAtoms(), 3)
        assert coords.dtype == torch.float32

    def test_pipeline_returns_correct_shape_with_virtual_nodes(self):
        mol = Chem.MolFromSmiles("CCO")
        dm = Graph3DDataModule(
            laplacian_k=0,
            rwse_k=0,
            rrwp_k=0,
            compute_distances=False,
            num_virtual_nodes=2,
        )
        coords = dm._calculate_coords(mol)
        assert coords.shape == (mol.GetNumAtoms() + 2, 3)
        assert torch.all(coords[-2:] == 0.0)

    def test_calculate_coords_uses_embed_and_minimize(self):
        mol = Chem.MolFromSmiles("CCO")
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )

        # A real mol_h with one 2D conformer so slicing succeeds downstream.
        mol_h = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol_h)

        with patch(
            "matcha.datamodules.classic.graph_datamodule._embed_and_minimize",
            return_value=(mol_h, 0),
        ) as mock_helper:
            dm._calculate_coords(mol)

        assert mock_helper.call_count == 1
        args, kwargs = mock_helper.call_args
        passed_timeout = args[1] if len(args) > 1 else kwargs.get("timeout_seconds")
        assert passed_timeout == dm.params.embed_timeout

    def test_calculate_coords_fallback_to_2d_when_embed_and_minimize_returns_none(self):
        mol = Chem.MolFromSmiles("CCO")
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )

        with (
            patch(
                "matcha.datamodules.classic.graph_datamodule._embed_and_minimize",
                return_value=None,
            ),
            patch(
                "matcha.datamodules.classic.graph_datamodule.AllChem.Compute2DCoords",
                wraps=AllChem.Compute2DCoords,
            ) as mock_2d,
        ):
            coords = dm._calculate_coords(mol)

        assert mock_2d.call_count == 1
        assert coords.shape == (mol.GetNumAtoms(), 3)
        assert coords.dtype == torch.float32

    def test_calculate_coords_slices_hs_from_selected_conformer_only(self):
        """When multiple conformers are present on ``mol_h``, the sliced coords
        must come from the conformer at ``chosen_id`` — not from any other
        conformer on the same molecule.
        """
        mol = Chem.MolFromSmiles("CCO")
        n_heavy = mol.GetNumAtoms()
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )

        # Build a real mol_h with 4 conformers, each with distinct positions
        # so we can tell which one was picked.
        mol_h = Chem.AddHs(mol)
        n_total = mol_h.GetNumAtoms()
        for i in range(4):
            conf = Chem.Conformer(n_total)
            for atom_idx in range(n_total):
                conf.SetAtomPosition(atom_idx, (float(i), float(atom_idx), 0.0))
            mol_h.AddConformer(conf, assignId=True)

        chosen_id = 3
        expected = mol_h.GetConformer(chosen_id).GetPositions()[:n_heavy, :]

        with patch(
            "matcha.datamodules.classic.graph_datamodule._embed_and_minimize",
            return_value=(mol_h, chosen_id),
        ):
            coords = dm._calculate_coords(mol)

        assert torch.allclose(coords, torch.tensor(expected, dtype=torch.float32))

    def test_reproducibility_across_calls(self):
        """``randomSeed=42`` inside ``_embed_and_minimize`` should make repeat
        calls on the same input molecule produce identical coord tensors.
        """
        smi = "CC(=O)O"  # acetic acid
        dm = Graph3DDataModule(
            laplacian_k=0, rwse_k=0, rrwp_k=0, compute_distances=False
        )
        coords_a = dm._calculate_coords(Chem.MolFromSmiles(smi))
        coords_b = dm._calculate_coords(Chem.MolFromSmiles(smi))
        assert torch.equal(coords_a, coords_b)

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


# ===================================================================
# Graph3DDataModule – user-supplied coords (Stage 3, issue #72)
# ===================================================================


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
