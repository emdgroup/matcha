"""Shared helpers for user-supplied 3D atomic coordinates.

Both :class:`Graph3DDataModule` (classic) and
:class:`Graph3DPretrainingDataModule` (pretraining) accept per-molecule 3D
coordinates supplied at featurize time. They live on different inheritance
branches, so validation and canonical-atom-order remapping are factored out
here as module-level helpers rather than hoisted onto a common base.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def validate_coords(
    mol_list: list[Mol],
    coords: list[np.ndarray],
) -> list[np.ndarray]:
    """Validate that user-supplied 3D coordinates match the molecules.

    Checks that:

    - ``len(coords) == len(mol_list)``
    - Each ``coords[i]`` has shape ``(A_i, 3)`` where ``A_i`` is the number
      of heavy atoms in the canonical SMILES of molecule *i*.
    - Each ``coords[i]`` contains only finite values (no NaN, no Inf).
      Non-finite entries would silently corrupt downstream distance
      features, so they are rejected up front.

    :param mol_list: list of RDKit molecules
    :param coords: list of per-molecule coord arrays
    :raises ValueError: on length, shape, atom-count, or finite-value
        violations
    :return: validated ``coords`` (each entry cast to ``float32`` ndarray)
    """
    if len(coords) != len(mol_list):
        raise ValueError(
            f"coords length ({len(coords)}) must match "
            f"mol_list length ({len(mol_list)})"
        )

    validated: list[np.ndarray] = []
    for i, (mol, ci) in enumerate(zip(mol_list, coords)):
        ci = np.asarray(ci, dtype=np.float32)
        if ci.ndim != 2 or ci.shape[1] != 3:
            raise ValueError(
                f"coords[{i}] must have shape (num_atoms, 3), "
                f"got shape {tuple(ci.shape)}"
            )

        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        expected_atoms = canonical_mol.GetNumAtoms()

        if ci.shape[0] != expected_atoms:
            raise ValueError(
                f"coords[{i}] has {ci.shape[0]} rows but molecule "
                f"has {expected_atoms} canonical atoms"
            )

        if not np.isfinite(ci).all():
            raise ValueError(f"coords[{i}] contains non-finite values (NaN or Inf)")

        validated.append(ci)
    return validated


def reorder_coords_to_canonical(
    mol: Mol,
    coords_i: np.ndarray,
) -> np.ndarray:
    """Reorder user-supplied coords to the canonical atom ordering.

    :meth:`GraphDataModule._calculate_graph` reparses the molecule via its
    canonical SMILES, so the atom indices on the resulting graph do not
    necessarily match the original ``mol``'s atom order. Coordinates
    supplied by the user in the original atom order therefore need to be
    remapped, otherwise ``graph.pos`` would be misaligned with ``graph.x``
    and ``y_node`` — a silent bug that would train the encoder on wrong
    atom-to-coord assignments.

    Uses ``mol.GetSubstructMatch(canonical_mol)`` to obtain, for each atom
    in the canonical mol, the index in the original mol.

    :param mol: RDKit molecule in user-supplied atom order
    :param coords_i: coordinate array of shape ``(A, 3)`` in the same atom
        order as ``mol``
    :raises ValueError: if the canonical mol cannot be matched back to the
        input atoms
    :return: coordinate array in canonical atom order
    """
    canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    match = mol.GetSubstructMatch(canonical_mol)
    if len(match) != canonical_mol.GetNumAtoms():
        raise ValueError(
            "Could not map canonical atom ordering back to the input mol; "
            "GetSubstructMatch returned an incomplete match."
        )
    return coords_i[list(match)]
