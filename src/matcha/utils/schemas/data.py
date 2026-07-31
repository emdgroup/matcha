"""Pydantic schemas and dataclasses for molecular datasets used in matcha."""

from matcha.utils.schemas.base import BaseDataModel
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol
from pydantic import field_validator, model_validator, ValidationInfo
import numpy as np
from dataclasses import dataclass


@dataclass
class MolReadout:
    """Dataclass representing a single molecule with its target value and censoring bound."""

    mol: Mol
    y: int | float | np.ndarray | None = None
    bound: str | list | None = None


class MolDataset(BaseDataModel):
    """Validated dataset of RDKit molecules with optional targets and censoring masks.

    Supports indexing and length queries, converting SMILES strings to Mol objects on
    construction.
    """

    mols: list[Mol] | list[str]
    y: np.ndarray | None = None
    bound_mask: list[list[str]] | list[str] | None = None

    @model_validator(mode="after")
    def check_lengths(self):
        """Validate that mols, y, and bound_mask have consistent lengths."""
        if self.y is not None and len(self.y) != len(self.mols):
            raise ValueError(
                f"The length of mols ({len(self.mols)}) and y ({len(self.y)}) must be the same."
            )

        if self.y is None and isinstance(self.bound_mask, list):
            raise ValueError(
                "If y is None, bound_mask must also be None, found list instead."
            )

        if self.bound_mask is not None:
            if isinstance(self.bound_mask[0], str) and len(self.bound_mask) != len(
                self.mols
            ):
                raise ValueError(
                    f"The length of mols ({len(self.mols)}) and bound_mask ({len(self.bound_mask)}) must be the same."
                )
            elif isinstance(self.bound_mask[0], list) and len(
                self.bound_mask[0]
            ) != len(self.mols):
                raise ValueError(
                    f"The length of mols ({len(self.mols)}) and bound_mask ({len(self.bound_mask[0])}) must be the same."
                )

        return self

    @field_validator("y")
    def validate_y(cls, v, info: ValidationInfo):
        """Validate that y is a 2D numpy array, reshaping 1D arrays as needed."""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("y must be a numpy vector")
            if len(v.shape) == 1:
                v = v.reshape(-1, 1)
            elif len(v.shape) != 2:
                raise ValueError(
                    f"y must be a two-dimensional array, instead got shape {v.shape}"
                )
        return v

    @field_validator("mols")
    def validate_mols(cls, v, info: ValidationInfo):
        """Validate and convert SMILES strings to RDKit Mol objects."""
        if not isinstance(v, list):
            raise ValueError("mols must be a list")

        if isinstance(v[0], str):
            v = [MolFromSmiles(x) for x in v]

        for mol_ith in v:
            if not isinstance(mol_ith, Mol):
                raise ValueError(
                    f"mols should be a list of RDKIT molecules, found {mol_ith}"
                )
        return v

    @field_validator("bound_mask")
    def validate_bound_mask(cls, v, info: ValidationInfo):
        """Validate that bound_mask is a list of strings or list of lists of strings."""
        if v is not None:
            if not isinstance(v[0], list):
                # Validate that v is a list of strings
                if not isinstance(v, list):
                    raise ValueError(f"{info.field_name} must be a list")
                for item in v:
                    if not isinstance(item, str):
                        raise ValueError(f"{info.field_name} must be a list of strings")
        return v

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        mol = self.mols[idx]
        y = self.y[idx] if self.y is not None else None
        if self.bound_mask is None:
            bound = None
        elif isinstance(self.bound_mask[0], str):
            bound = self.bound_mask[idx]
        elif isinstance(self.bound_mask[0], list):
            bound = [x[idx] for x in self.bound_mask]
        return MolReadout(mol, y, bound)
