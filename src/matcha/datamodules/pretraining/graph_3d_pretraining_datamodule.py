"""3D graph pretraining DataModule with user-supplied atomic coordinates.

Extends :class:`GraphPretrainingDataModule` with a required ``coords`` argument
supplied by the user at featurize time. Coordinates are validated against the
canonical-SMILES atom count, reordered to the canonical atom ordering that
:meth:`GraphDataModule._calculate_graph` produces, and attached to each
per-molecule :class:`torch_geometric.data.Data` on ``pos`` — the PyG
convention. ``Batch.from_data_list`` then auto-concatenates ``pos`` alongside
``x``, ``y_node`` and the positional encodings, so the pretraining collate
function inherited from the parent needs no changes and the encoder reads
coords from ``graph.pos`` inside its per-layer hook.

ETKDG-style on-the-fly conformer generation is never performed inside this
datamodule.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from torch.utils.data import StackDataset
from torch_geometric.data import Data

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.classic.graph_datamodule import Graph3DDataModule
from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)
from matcha.utils.schemas.datamodules import Graph3DPretrainingDataModuleInputModel
from matcha.utils.wrapper import parallelize


@DataModuleRegistry.register("graph3d_pretraining")
class Graph3DPretrainingDataModule(GraphPretrainingDataModule):
    """3D graph pretraining datamodule with user-supplied atomic coordinates.

    Extends :class:`GraphPretrainingDataModule` with a required ``coords``
    argument supplied at featurize time. Per-molecule 3D coordinates ride on
    ``Data.pos`` so that :class:`torch_geometric.data.Batch` auto-concatenates
    them alongside ``x``, ``y_node`` and the positional encodings — the parent
    collate function needs no changes.

    Example usage:

    .. code-block:: python

        dm = Graph3DPretrainingDataModule()

        # coords is a list of arrays, one per molecule, each of shape (num_atoms, 3)
        # y_node is a list of arrays, one per molecule, each of shape (num_atoms, T)
        # y_graph is a numpy array of shape (N, G)
        dataset = dm.featurize(
            mol_list=mols,
            y_graph=y_graph,
            y_node=y_node,
            coords=coords,
        )

    All constructor parameters are inherited from
    :class:`GraphPretrainingDataModule`.
    """

    def __init__(
        self,
        scale_y_graph: bool = False,
        scale_y_node: bool = False,
        laplacian_k: int = 10,
        rwse_k: int = 20,
        elstatic_k: int = 0,
        distmat_k: int = 0,
        rrwp_k: int = 20,
        compute_distances: bool = True,
        num_virtual_nodes: int = 0,
        init_virtual_nodes: bool = False,
        batch_size: int = 256,
        num_workers: int = 0,
        augment_resonance: bool = False,
    ):
        """Initialise the 3D graph pretraining datamodule.

        :param scale_y_graph: whether to standardise molecule-level targets
        :param scale_y_node: whether to standardise atom-level targets
        :param laplacian_k: number of Laplacian positional encoding dimensions
        :param rwse_k: number of random-walk structural encoding dimensions
        :param elstatic_k: number of electrostatic positional encoding dimensions
        :param distmat_k: number of distance-matrix positional encoding dimensions
        :param rrwp_k: number of relative random-walk probability dimensions
        :param compute_distances: whether to compute interatomic distances
        :param num_virtual_nodes: number of virtual nodes to add
        :param init_virtual_nodes: whether to initialise virtual node features
        :param batch_size: training batch size
        :param num_workers: number of dataloader workers
        :param augment_resonance: whether to apply resonance augmentation
        """
        super().__init__(
            scale_y_graph=scale_y_graph,
            scale_y_node=scale_y_node,
            laplacian_k=laplacian_k,
            rwse_k=rwse_k,
            elstatic_k=elstatic_k,
            distmat_k=distmat_k,
            rrwp_k=rrwp_k,
            compute_distances=compute_distances,
            num_virtual_nodes=num_virtual_nodes,
            init_virtual_nodes=init_virtual_nodes,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )

        # Swap the parent's schema for the 3D pretraining variant so
        # ``datamodule_type`` reflects the correct discriminator.
        self.params = Graph3DPretrainingDataModuleInputModel(
            scale_y_graph=scale_y_graph,
            scale_y_node=scale_y_node,
            laplacian_k=laplacian_k,
            rwse_k=rwse_k,
            elstatic_k=elstatic_k,
            distmat_k=distmat_k,
            rrwp_k=rrwp_k,
            compute_distances=compute_distances,
            num_virtual_nodes=num_virtual_nodes,
            init_virtual_nodes=init_virtual_nodes,
            is_classification=False,
            scaler_type="standard",
            clip=False,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )

    # ------------------------------------------------------------------
    # Export to classic
    # ------------------------------------------------------------------

    def export_to_classic(self) -> Graph3DDataModule:
        """Return a :class:`Graph3DDataModule` that mirrors the current state.

        The exported instance inherits the positional-encoding settings from
        the pretraining datamodule so that downstream ``E3GNNModel``
        finetuning uses a matching featurization. ``embed_timeout`` is left at
        its :class:`Graph3DDataModule` default — the pretraining variant never
        runs ETKDG, so it has no timeout parameter of its own.

        :return Graph3DDataModule: a classic 3D graph datamodule with the
            same positional-encoding config
        """
        p = self.params
        return Graph3DDataModule(
            laplacian_k=p.laplacian_k,
            rwse_k=p.rwse_k,
            elstatic_k=p.elstatic_k,
            distmat_k=p.distmat_k,
            rrwp_k=p.rrwp_k,
            compute_distances=p.compute_distances,
            num_virtual_nodes=p.num_virtual_nodes,
            init_virtual_nodes=p.init_virtual_nodes,
            is_classification=p.is_classification,
            scaler_type=p.scaler_type,
            clip=p.clip,
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            augment_resonance=p.augment_resonance,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_coords(
        self,
        mol_list: list[Mol],
        coords: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Validate that user-supplied 3D coordinates match the molecules.

        Checks that:
        - ``len(coords) == len(mol_list)``
        - Each ``coords[i]`` has shape ``(A_i, 3)`` where ``A_i`` is the
          number of heavy atoms in the canonical SMILES of molecule *i*.

        :param mol_list: list of RDKit molecules
        :param coords: list of per-molecule coord arrays
        :raises ValueError: on length or shape mismatch
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
            validated.append(ci)
        return validated

    # ------------------------------------------------------------------
    # Coord reordering + graph construction helper
    # ------------------------------------------------------------------

    def _reorder_coords_to_canonical(
        self,
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
        :param coords_i: coordinate array of shape ``(A, 3)`` in the same
            atom order as ``mol``
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

    def _calculate_graph_with_node_labels_and_pos(
        self,
        mol: Mol,
        y_node_i: np.ndarray,
        coords_i: np.ndarray,
    ) -> Data:
        """Build a PyG graph, attach atom-level labels, then attach 3D coords.

        Delegates the graph construction and ``y_node`` handling to the parent
        :meth:`GraphPretrainingDataModule._calculate_graph_with_node_labels`.
        Then reorders ``coords_i`` to the canonical atom ordering, zero-pads
        for virtual nodes (matching the existing :class:`Graph3DDataModule`
        convention — NaN pads would poison E3GNN's Fourier distance features
        on real neighbours of a virtual node), and attaches the result to
        ``graph.pos``.

        :param mol: RDKit molecule
        :param y_node_i: array of shape ``(A_i, T)`` with atom labels in
            canonical atom order
        :param coords_i: array of shape ``(A_i, 3)`` with 3D coordinates in
            the input mol's atom order
        :return: PyG ``Data`` object with ``y_node`` and ``pos`` attached
        """
        graph = self._calculate_graph_with_node_labels(mol, y_node_i)

        canonical_coords = self._reorder_coords_to_canonical(mol, coords_i)
        pos = torch.tensor(canonical_coords, dtype=torch.float32)

        if self.params.num_virtual_nodes > 0:
            pad = torch.zeros(
                (self.params.num_virtual_nodes, 3),
                dtype=torch.float32,
            )
            pos = torch.cat([pos, pad], dim=0)

        graph.pos = pos
        return graph

    def _process_batch_with_node_labels_and_pos(
        self,
        batch: list[tuple[Mol, np.ndarray, np.ndarray]],
    ) -> list[Data]:
        """Process a batch of ``(mol, y_node, coords)`` triples into PyG graphs.

        :param batch: list of ``(Mol, y_node_i, coords_i)`` tuples
        :return: list of PyG ``Data`` objects with ``y_node`` and ``pos``
        """
        return [
            self._calculate_graph_with_node_labels_and_pos(mol, yn, ci)
            for mol, yn, ci in batch
        ]

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def generate_features(
        self,
        mol_list: list[Mol],
        y_graph: np.ndarray | None = None,
        y_node: list[np.ndarray] | None = None,
        coords: list[np.ndarray] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generate unscaled features with node labels and 3D coordinates.

        :param mol_list: list of N RDKit molecules
        :param y_graph: array ``(N, G)`` of molecule-level targets, or None
        :param y_node: list of N arrays, each ``(A_i, T)`` of atom-level
            targets in canonical atom order
        :param coords: list of N arrays, each ``(A_i, 3)`` of 3D coordinates
            in the input mol's atom order (reordered internally)
        :param n_jobs: number of parallel workers (None = auto)
        :return: ``StackDataset`` with keys ``graph`` and ``y_graph``. Each
            ``graph`` has ``y_node`` and ``pos`` attributes.
        """
        mol_list, y_graph, _, n_jobs = self._validate_input(
            mol_list, y_graph, None, n_jobs
        )

        if y_node is None:
            raise ValueError("y_node must be provided for graph pretraining")
        if coords is None:
            raise ValueError(
                "coords must be provided for 3D graph pretraining; "
                "this datamodule does not compute conformers on the fly"
            )

        y_node = self._validate_node_labels(mol_list, y_node)
        coords = self._validate_coords(mol_list, coords)

        triples = list(zip(mol_list, y_node, coords))

        graphs = parallelize(
            self._process_batch_with_node_labels_and_pos,
            triples,
            n_jobs,
        )

        y_graph_tensor = torch.tensor(y_graph, dtype=torch.float32)
        return StackDataset(graph=graphs, y_graph=y_graph_tensor)

    # ------------------------------------------------------------------
    # Featurize (main entry-point)
    # ------------------------------------------------------------------

    def featurize(
        self,
        mol_list: list[Mol],
        y_graph: np.ndarray | None = None,
        y_node: list[np.ndarray] | None = None,
        coords: list[np.ndarray] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generate a dataset ready for 3D graph pretraining.

        :param mol_list: list of N RDKit molecules
        :param y_graph: array ``(N, G)`` of molecule-level targets
        :param y_node: list of N arrays, each ``(A_i, T)`` of atom-level
            targets in canonical atom order
        :param coords: list of N arrays, each ``(A_i, 3)`` of 3D coordinates
            in the input mol's atom order
        :param is_training: whether to fit the Y scaler
        :param n_jobs: number of parallel workers (None = auto)
        :return: ``StackDataset`` with keys ``graph`` and ``y_graph``
        """
        dataset = self.generate_features(mol_list, y_graph, y_node, coords, n_jobs)

        if self.params.scale_y_graph:
            if is_training:
                self._fit_y_graph(dataset)
            self._transform_y_graph(dataset)

        if self.params.scale_y_node:
            if is_training:
                self._fit_y_node(dataset)
            self._transform_y_node(dataset)

        return dataset

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise state for MLFlow logging.

        :return: dict containing ID, params, and fitted scalers
        """
        state = {
            "ID": "graph3d_pretraining",
            "params": self.params.model_dump(),
            "augment_resonance": self._augment_resonance,
        }
        if self.params.scale_y_graph:
            state["y_scaler"] = self._y_scaler
        if self.params.scale_y_node:
            state["y_node_scaler"] = self._y_node_scaler
        return state

    def load_state_dict(self, state_dict: dict):
        """Restore state from a previously serialised dict.

        :param state_dict: dict produced by :meth:`state_dict`
        """
        self.params = Graph3DPretrainingDataModuleInputModel(**state_dict["params"])
        self._augment_resonance = state_dict.get("augment_resonance", False)
        if "y_scaler" in state_dict:
            self._y_scaler = state_dict["y_scaler"]
        if "y_node_scaler" in state_dict:
            self._y_node_scaler = state_dict["y_node_scaler"]

    @classmethod
    def dummy(cls):
        """Create a dummy instance with default parameters.

        :return: a new :class:`Graph3DPretrainingDataModule` with default settings
        """
        return cls()
