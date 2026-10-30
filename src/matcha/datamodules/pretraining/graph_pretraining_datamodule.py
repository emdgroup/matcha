"""Graph pretraining DataModule for supervised multi-task learning on graphs.

Provides a datamodule that extends GraphDataModule to handle both molecule-level
and atom-level labels for pretraining graph neural networks such as GIN. The user
supplies external per-atom labels (e.g. partial charges, electronegativity, or any
computed atom-level property) alongside molecule-level labels. The model then
learns to predict both simultaneously.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from torch.utils.data import StackDataset
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.classic.graph_datamodule import GraphDataModule
from matcha.datamodules.utils import collate_fn_pyg_graph
from matcha.utils.schemas.datamodules import GraphPretrainingDataModuleInputModel
from matcha.utils.wrapper import parallelize


@DataModuleRegistry.register("graph_pretraining")
class GraphPretrainingDataModule(GraphDataModule):
    """Graph DataModule for multi-task pretraining with atom-level and molecule-level labels.

    Extends :class:`GraphDataModule` to accept user-provided per-atom labels
    alongside molecule-level labels. This enables supervised pretraining of graph
    neural networks (e.g. :class:`GINPretraining`) that jointly predict atom-level
    targets (such as partial charges, SASA, electronegativity, etc.) and
    molecule-level targets (such as logP, molecular weight, etc.).

    The atom-level labels are **not** derived from the graph's own atom features —
    they are externally computed properties supplied by the user.

    Example usage:

    .. code-block:: python

        dm = GraphPretrainingDataModule()

        # y_node is a list of arrays, one per molecule, each of shape (num_atoms, T)
        # y_graph is a numpy array of shape (N, G)
        dataset = dm.featurize(
            mol_list=mols,
            y_graph=y_graph,
            y_node=y_node,
        )

    :param bool scale_y_graph: whether to fit a scaler on the molecule-level
        targets during training, defaults to False

    All other parameters are inherited from :class:`GraphDataModule`.
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
        """Initialise the graph pretraining datamodule.

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
        # Pretraining always uses regression-style targets, no classification
        # encoding or label transforms.
        super().__init__(
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
            label_encoder_params={},
            label_transform_params={},
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )

        self._y_node_scaler = StandardScaler()

        # Override params with pretraining-specific schema
        self.params = GraphPretrainingDataModuleInputModel(
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

    def export_to_classic(self) -> GraphDataModule:
        """Return a :class:`GraphDataModule` that mirrors the current state.

        The exported instance inherits all graph-specific settings
        (positional encoding dimensions, virtual nodes, etc.) so that it can
        be used directly for downstream (non-pretraining) training or
        inference.

        :return GraphDataModule: a classic graph datamodule with the same state
        """
        p = self.params
        dm = GraphDataModule(
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
        return dm

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_node_labels(
        self,
        mol_list: list[Mol],
        y_node: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Validate that atom-level labels match the molecules.

        Checks that:
        - ``len(y_node) == len(mol_list)``
        - Each ``y_node[i]`` has ``A_i`` rows where ``A_i`` is the
          number of heavy atoms in molecule *i*.

        :param mol_list: list of RDKit molecules
        :param y_node: list of per-atom label arrays
        :raises ValueError: on shape mismatch
        :return: validated ``y_node``
        """
        if len(y_node) != len(mol_list):
            raise ValueError(
                f"y_node length ({len(y_node)}) must match "
                f"mol_list length ({len(mol_list)})"
            )

        for i, (mol, yn) in enumerate(zip(mol_list, y_node)):
            yn = np.asarray(yn)
            if yn.ndim == 1:
                yn = yn.reshape(-1, 1)
                y_node[i] = yn

            # Atom count is determined from canonical SMILES (same as
            # _calculate_graph which re-parses the canonical SMILES).
            canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
            expected_atoms = canonical_mol.GetNumAtoms()

            if yn.shape[0] != expected_atoms:
                raise ValueError(
                    f"y_node[{i}] has {yn.shape[0]} rows but molecule "
                    f"has {expected_atoms} atoms"
                )
        return y_node

    # ------------------------------------------------------------------
    # Graph construction helper
    # ------------------------------------------------------------------

    def _calculate_graph_with_node_labels(
        self,
        mol: Mol,
        y_node_i: np.ndarray,
    ) -> Data:
        """Build a PyG graph and attach externally-provided atom-level labels.

        Calls the parent :meth:`_calculate_graph` then stores the per-atom
        labels as ``graph.y_node``.  If virtual nodes are enabled, the labels
        are zero-padded for the virtual nodes.

        :param mol: RDKit molecule
        :param y_node_i: array of shape ``(A_i, T)`` with atom labels
        :return: PyG ``Data`` object with ``y_node`` attribute
        """
        graph = self._calculate_graph(mol)

        y_node_tensor = torch.tensor(y_node_i, dtype=torch.float32)

        # Pad with NaN for virtual nodes so they are masked out in loss
        if self.params.num_virtual_nodes > 0:
            pad = torch.full(
                (self.params.num_virtual_nodes, y_node_tensor.shape[1]),
                float("nan"),
                dtype=torch.float32,
            )
            y_node_tensor = torch.cat([y_node_tensor, pad], dim=0)

        graph.y_node = y_node_tensor
        return graph

    def _process_batch_with_node_labels(
        self,
        batch: list[tuple[Mol, np.ndarray]],
    ) -> list[Data]:
        """Process a batch of (molecule, node_labels) pairs into PyG graphs.

        :param batch: list of ``(Mol, y_node_i)`` tuples
        :return: list of PyG ``Data`` objects with ``y_node`` attached
        """
        return [self._calculate_graph_with_node_labels(mol, yn) for mol, yn in batch]

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def generate_features(
        self,
        mol_list: list[Mol],
        y_graph: np.ndarray | None = None,
        y_node: list[np.ndarray] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generate unscaled graph features with atom-level and molecule-level labels.

        :param mol_list: list of N RDKit molecules
        :param y_graph: array ``(N, G)`` of molecule-level targets, or None
        :param y_node: list of N arrays, each ``(A_i, T)`` of atom-level targets,
            or None
        :param n_jobs: number of parallel workers (None = auto)
        :return: ``StackDataset`` with keys ``graph`` and ``y_graph``
            (atom-level labels are stored on each ``Data.y_node``)
        """
        # Use base validation for mol_list / y_graph
        mol_list, y_graph, _, n_jobs = self._validate_input(
            mol_list, y_graph, None, n_jobs
        )

        if y_node is None:
            raise ValueError("y_node must be provided for graph pretraining")

        y_node = self._validate_node_labels(mol_list, y_node)

        # Build (mol, y_node_i) pairs and parallelise
        mol_yn_pairs = list(zip(mol_list, y_node))

        graphs = parallelize(
            self._process_batch_with_node_labels,
            mol_yn_pairs,
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
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generate a dataset ready for graph pretraining.

        Processes molecules alongside their molecule-level and atom-level
        labels into a ``StackDataset`` that can be consumed by
        :class:`BaseGraphPretrainingModel`.

        :param mol_list: list of N RDKit molecules
        :param y_graph: array ``(N, G)`` of molecule-level targets
        :param y_node: list of N arrays, each ``(A_i, T)`` of atom-level targets
        :param is_training: whether to fit the Y scaler (only affects
            ``y_graph`` scaling when ``scale_y_graph=True``)
        :param n_jobs: number of parallel workers (None = auto)
        :return: ``StackDataset`` with keys ``graph`` and ``y_graph``
        """
        dataset = self.generate_features(mol_list, y_graph, y_node, n_jobs)

        # Optionally scale molecule-level labels
        if self.params.scale_y_graph:
            if is_training:
                self._fit_y_graph(dataset)
            self._transform_y_graph(dataset)

        # Optionally scale atom-level labels
        if self.params.scale_y_node:
            if is_training:
                self._fit_y_node(dataset)
            self._transform_y_node(dataset)

        return dataset

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    def _fit_y_graph(self, dataset: StackDataset) -> None:
        """Fit the Y scaler on molecule-level targets.

        :param dataset: StackDataset containing a ``y_graph`` tensor
        """
        y = dataset.datasets["y_graph"].numpy().copy()
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self._y_scaler.fit(y)

    def _transform_y_graph(self, dataset: StackDataset) -> None:
        """Transform molecule-level targets using the fitted scaler.

        :param dataset: StackDataset whose ``y_graph`` tensor is scaled in-place
        """
        y = dataset.datasets["y_graph"].numpy().copy()
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if hasattr(self._y_scaler, "n_features_in_"):
            y = self._y_scaler.transform(y)
        dataset.datasets["y_graph"] = torch.tensor(y, dtype=torch.float32)

    def _fit_y_node(self, dataset: StackDataset) -> None:
        """Fit the node scaler on atom-level targets across all graphs.

        :param dataset: StackDataset containing graphs with ``y_node`` attributes
        """
        graphs = dataset.datasets["graph"]
        all_y_node = []
        for g in graphs:
            yn = g.y_node.numpy()
            all_y_node.append(yn)
        y = np.concatenate(all_y_node, axis=0)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._y_node_scaler.fit(y)

    def _transform_y_node(self, dataset: StackDataset) -> None:
        """Transform atom-level targets in-place using the fitted node scaler.

        :param dataset: StackDataset containing graphs with ``y_node`` attributes
        """
        if not hasattr(self._y_node_scaler, "n_features_in_"):
            return
        graphs = dataset.datasets["graph"]
        for g in graphs:
            yn = g.y_node.numpy()
            if yn.ndim == 1:
                yn = yn.reshape(-1, 1)
            yn = self._y_node_scaler.transform(yn)
            g.y_node = torch.tensor(yn, dtype=torch.float32)

    def fit(self, dataset: StackDataset) -> None:
        """Fit scalers on labels if scaling is enabled.

        :param dataset: StackDataset produced by :meth:`generate_features`
        """
        if self.params.scale_y_graph:
            self._fit_y_graph(dataset)
        if self.params.scale_y_node:
            self._fit_y_node(dataset)

    def transform(self, dataset: StackDataset) -> StackDataset:
        """Scale labels if scaling is enabled.

        :param dataset: StackDataset to transform in-place
        :return: the same dataset with scaled labels
        """
        if self.params.scale_y_graph:
            self._transform_y_graph(dataset)
        if self.params.scale_y_node:
            self._transform_y_node(dataset)
        return dataset

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def collate_fn(self, data: list[dict]) -> dict:
        """Collate a list of samples into a pretraining batch.

        Produces the batch format expected by
        :class:`BaseGraphPretrainingModel`:

        - ``graph``: batched PyG graph (``y_node`` is auto-concatenated
          by ``Batch.from_data_list`` as a node-level attribute)
        - ``y_node``: ``[total_nodes_in_batch, T]``
        - ``y_graph``: ``[batch_size, G]``

        :param data: list of dicts from the ``StackDataset``
        :return: dict with keys ``graph``, ``y_node``, ``y_graph``
        """
        dict_of_lists = {k: [d[k] for d in data] for k in data[0]}
        graphs = dict_of_lists["graph"]
        y_graph_list = dict_of_lists["y_graph"]

        # Batch graphs — PyG auto-concatenates node-level attrs (including y_node)
        bg = collate_fn_pyg_graph(graphs)

        # Extract concatenated y_node from the batched graph
        y_node = bg.y_node
        # Remove from batched graph to keep the graph payload clean
        del bg.y_node

        y_graph = torch.stack(y_graph_list, dim=0)

        return {"graph": bg, "y_node": y_node, "y_graph": y_graph}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise state for MLFlow logging.

        :return: dict containing ID, params, and fitted scalers
        """
        state = {
            "ID": "graph_pretraining",
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
        self.params = GraphPretrainingDataModuleInputModel(**state_dict["params"])
        self._augment_resonance = state_dict.get("augment_resonance", False)
        if "y_scaler" in state_dict:
            self._y_scaler = state_dict["y_scaler"]
        if "y_node_scaler" in state_dict:
            self._y_node_scaler = state_dict["y_node_scaler"]

    @classmethod
    def dummy(cls):
        """Create a dummy instance with default parameters.

        :return: a new :class:`GraphPretrainingDataModule` with default settings
        """
        return cls()
