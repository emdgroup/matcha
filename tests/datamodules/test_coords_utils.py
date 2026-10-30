"""Unit tests for the shared 3D-coord helpers in ``coords_utils``."""

import numpy as np
import pytest
from rdkit import Chem

from matcha.datamodules.classic.coords_utils import (
    reorder_coords_to_canonical,
    validate_coords,
)


# ===================================================================
# Helpers
# ===================================================================


def _canonical_num_atoms(mol) -> int:
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    return canonical.GetNumAtoms()


def _distinct_coords(n_atoms: int) -> np.ndarray:
    """Coords where row i = [i, i+100, i+200] — each row is unique."""
    i = np.arange(n_atoms, dtype=np.float32)
    return np.stack([i, i + 100.0, i + 200.0], axis=1)


# ===================================================================
# validate_coords
# ===================================================================


class TestValidateCoords:
    def test_happy_path_cast_to_float32(self, small_mol_list):
        # Feed float64 to prove the cast happens.
        coords = [
            np.zeros((_canonical_num_atoms(m), 3), dtype=np.float64)
            for m in small_mol_list
        ]
        validated = validate_coords(small_mol_list, coords)
        assert len(validated) == len(small_mol_list)
        for ci, mol in zip(validated, small_mol_list):
            assert ci.shape == (_canonical_num_atoms(mol), 3)
            assert ci.dtype == np.float32

    def test_length_mismatch_raises(self, small_mol_list):
        coords = [
            np.zeros((_canonical_num_atoms(m), 3), dtype=np.float32)
            for m in small_mol_list[:2]
        ]
        with pytest.raises(ValueError, match="coords length"):
            validate_coords(small_mol_list, coords)

    def test_wrong_ndim_raises(self, small_mol_list):
        coords = [np.zeros(3, dtype=np.float32) for _ in small_mol_list]
        with pytest.raises(ValueError, match=r"shape \(num_atoms, 3\)"):
            validate_coords(small_mol_list, coords)

    def test_wrong_last_dim_raises(self, small_mol_list):
        coords = [
            np.zeros((_canonical_num_atoms(m), 2), dtype=np.float32)
            for m in small_mol_list
        ]
        with pytest.raises(ValueError, match=r"shape \(num_atoms, 3\)"):
            validate_coords(small_mol_list, coords)

    def test_atom_count_mismatch_raises(self, small_mol_list):
        coords = [
            np.zeros((_canonical_num_atoms(m) + 1, 3), dtype=np.float32)
            for m in small_mol_list
        ]
        with pytest.raises(ValueError, match="rows but molecule"):
            validate_coords(small_mol_list, coords)

    def test_nan_raises_with_mol_index(self, small_mol_list):
        coords = [
            np.zeros((_canonical_num_atoms(m), 3), dtype=np.float32)
            for m in small_mol_list
        ]
        # Poison the third molecule so the error message identifies it.
        coords[2][0, 0] = np.nan
        with pytest.raises(ValueError, match=r"coords\[2\].*non-finite"):
            validate_coords(small_mol_list, coords)

    def test_inf_raises_with_mol_index(self, small_mol_list):
        coords = [
            np.zeros((_canonical_num_atoms(m), 3), dtype=np.float32)
            for m in small_mol_list
        ]
        coords[1][2, 1] = np.inf
        with pytest.raises(ValueError, match=r"coords\[1\].*non-finite"):
            validate_coords(small_mol_list, coords)


# ===================================================================
# reorder_coords_to_canonical
# ===================================================================


class TestReorderCoordsToCanonical:
    def test_identity_on_canonical_input(self):
        # Building the mol from the canonical SMILES means the input atom
        # order already matches the canonical order, so the reorder is a
        # no-op.
        smi = Chem.MolToSmiles(Chem.MolFromSmiles("OCC(=O)N"), canonical=True)
        mol = Chem.MolFromSmiles(smi)
        coords = _distinct_coords(mol.GetNumAtoms())

        reordered = reorder_coords_to_canonical(mol, coords)

        np.testing.assert_array_equal(reordered, coords)

    def test_permutation_when_input_differs_from_canonical(self):
        # A non-canonical SMILES so the input atom order differs from
        # the canonical order — the reorder must apply the substruct-match
        # permutation.
        smi = "OCC(=O)N"  # glycolamide — chosen for a non-trivial canonicalisation
        mol = Chem.MolFromSmiles(smi)
        n = mol.GetNumAtoms()
        coords = _distinct_coords(n)

        reordered = reorder_coords_to_canonical(mol, coords)

        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        match = mol.GetSubstructMatch(canonical_mol)

        # Sanity: this fixture must actually exercise a non-identity
        # permutation, otherwise the assertion below is vacuous.
        assert list(match) != list(range(n))

        # Row i of the reordered coords comes from row match[i] of the input.
        for i, m in enumerate(match):
            np.testing.assert_allclose(reordered[i], coords[m])
