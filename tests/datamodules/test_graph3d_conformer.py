"""Tests for Graph3D conformer generation and selection."""

from unittest.mock import MagicMock, patch

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from matcha.datamodules.classic.graph_datamodule import (
    Graph3DDataModule,
    _embed_and_minimize,
    _select_lowest_energy_conf_id,
)


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
        assert mmff_kwargs["maxIters"] == 500
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
