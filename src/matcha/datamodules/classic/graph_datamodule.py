"""Graph-based molecular featurization for 2D and 3D GNN training."""

import numpy as np
import torch
from chemprop import featurizers
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Data
from torch.utils.data import StackDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from matcha.datamodules.base_datamodule import BaseDataModule, DataModuleRegistry
from matcha.datamodules.classic.graph_positional_encoder import GraphPE
from matcha.utils.schemas.datamodules import (
    Graph3DDataModuleInputModel,
    GraphDataModuleInputModel,
)
from matcha.datamodules.utils import collate_fn_pyg_graph
from matcha.utils.wrapper import parallelize

np.random.seed(0)

ATOM_FEAT_DIM = 72
BOND_FEAT_DIM = 14


def _embed_with_timeout(mol: Mol, etkdg_params, timeout_seconds: float) -> int:
    """Run AllChem.EmbedMolecule in a thread with a timeout.

    RDKit releases the GIL during its C++ distance geometry solver, so a
    ThreadPoolExecutor future can interrupt a hung call from within a worker.
    Using shutdown(wait=False) ensures we do not block on stuck C++ threads
    after the timeout fires.

    Returns 0 on success, -1 on timeout or any exception.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(AllChem.EmbedMolecule, mol, etkdg_params)
    try:
        return future.result(timeout=timeout_seconds)
    except (FuturesTimeoutError, Exception):
        return -1
    finally:
        executor.shutdown(wait=False)


@DataModuleRegistry.register("graph")
class GraphDataModule(BaseDataModule):
    """2D Graph molecular representation featurization class using PyTorch Geometric.

    Allows users to convert a list of rdkit molecules and labels into a format ready
    to be used for graph neural network training. It uses the Chemprop featurizer
    backend to compute molecular graphs and converts them to PyTorch Geometric format.
    Common featurization logic is inherited from :class:`BaseDataModule`.

    The main purpose of the class is to enable the use of :method:`featurize`.
    Please check out :method:`featurize` for further information on the class' usage.

    :param int laplacian_k: number of components to use for the Laplacian PE embedding,
        defaults to 10. If set to 0, none are computed.

    :param int rwse_k: number of steps for Random Walk Structural Encoding,
        defaults to 20. If set to 0, none are computed.

    :param int elstatic_k: number of electrostatic PE components,
        defaults to 0.

    :param int distmat_k: number of distance matrix PE components,
        defaults to 0.

    :param int rrwp_k: number of relative random walk probability components,
        defaults to 20. If set to 0, none are computed.

    :param bool compute_distances: whether to compute shortest path distances,
        defaults to True.

    :param int num_virtual_nodes: number of virtual nodes to add to the graph to
        improve message passing, defaults to 0.

    :param bool init_virtual_nodes: whether to initialize virtual nodes with molecular
        descriptors, defaults to False.
    """

    def __init__(
        self,
        laplacian_k: int = 10,
        rwse_k: int = 20,
        elstatic_k: int = 0,
        distmat_k: int = 0,
        rrwp_k: int = 20,
        compute_distances: bool = True,
        num_virtual_nodes: int = 0,
        init_virtual_nodes: bool = False,
        is_classification: bool = False,
        scaler_type: str = "standard",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_params: dict = {},
        batch_size: int = 256,
        num_workers: int = 0,
        augment_resonance: bool = False,
    ):
        self.params = GraphDataModuleInputModel(
            laplacian_k=laplacian_k,
            rwse_k=rwse_k,
            elstatic_k=elstatic_k,
            distmat_k=distmat_k,
            rrwp_k=rrwp_k,
            compute_distances=compute_distances,
            num_virtual_nodes=num_virtual_nodes,
            init_virtual_nodes=init_virtual_nodes,
            is_classification=is_classification,
            scaler_type=scaler_type,
            clip=clip,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )

        super().__init__(
            scaler_type=scaler_type,
            augment_resonance=augment_resonance,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
        )

        # Initialize chemprop featurizer
        self._chemprop_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=featurizers.atom.MultiHotAtomFeaturizer.v2(),
            bond_featurizer=featurizers.bond.MultiHotBondFeaturizer(),
        )
        # Update collate function map for PyG Data objects
        self.collate_fn_map.update({Data: collate_fn_pyg_graph})

    def _calculate_graph(self, mol: Mol, is_training: bool = True) -> Data:
        """Converts an RDKit molecule into a PyTorch Geometric Data object.

        Uses the Chemprop featurizer for node and edge features, then converts
        to PyG format and adds positional encodings.

        :param Mol mol: molecule to convert
        :param bool is_training: whether this is for training (unused, for compatibility)

        :return Data: PyTorch Geometric graph ready for GNN input
        """
        # Sort atoms according to canonical SMILES order
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))

        # Use chemprop featurizer to get molecular graph features
        molgraph = self._chemprop_featurizer(mol)

        # Convert to PyG Data object
        # mg.edge_index is (2, num_edges) numpy array
        # mg.V is (num_atoms, atom_feat_dim) numpy array for node features
        # mg.E is (num_edges, edge_feat_dim) numpy array for edge features
        edge_index = torch.from_numpy(molgraph.edge_index).long()
        x = torch.from_numpy(molgraph.V).float()
        edge_attr = torch.from_numpy(molgraph.E).float()

        num_nodes = x.shape[0]

        # Create the PyG Data object (without virtual nodes first for PE computation)
        graph = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes
        )

        # Compute positional encodings BEFORE adding virtual nodes
        # so that PEs are not affected by virtual node connectivity
        if self.params.laplacian_k > 0:
            graph.laplacian_k = GraphPE.lap_pe(graph, self.params.laplacian_k)

        if self.params.rwse_k > 0:
            graph.rwse_k = GraphPE.rw_se(graph, self.params.rwse_k)

        if self.params.elstatic_k > 0:
            graph.elstatic_k = GraphPE.elstatic_pe(mol, self.params.elstatic_k)

        if self.params.distmat_k > 0:
            graph.distmat_k = GraphPE.distmat_pe(mol, self.params.distmat_k)

        if self.params.rrwp_k > 0:
            graph.rrwp_k = GraphPE.rrwp_re(graph, self.params.rrwp_k)

        if self.params.compute_distances:
            graph.spd = self._compute_shortest_paths(graph, num_nodes)
        else:
            graph.spd = None

        # Now add virtual nodes AFTER positional encoding computation
        if self.params.num_virtual_nodes > 0:
            # Add virtual node indicator feature to real nodes (0 = real)
            real_node_indicator = torch.zeros((num_nodes, 1), dtype=torch.float32)
            graph.x = torch.cat([graph.x, real_node_indicator], dim=1)

            # Add virtual edge indicator feature to real edges (0 = real)
            num_real_edges = edge_attr.shape[0] if edge_attr.numel() > 0 else 0
            real_edge_indicator = torch.zeros((num_real_edges, 1), dtype=torch.float32)
            graph.edge_attr = torch.cat([graph.edge_attr, real_edge_indicator], dim=1)

            # Add virtual nodes with zero features + virtual indicator (1 = virtual)
            virtual_node_feats = torch.zeros(
                (self.params.num_virtual_nodes, x.shape[1]), dtype=torch.float32
            )
            virtual_node_indicator = torch.ones(
                (self.params.num_virtual_nodes, 1), dtype=torch.float32
            )
            virtual_node_feats = torch.cat(
                [virtual_node_feats, virtual_node_indicator], dim=1
            )
            graph.x = torch.cat([graph.x, virtual_node_feats], dim=0)

            # Add edges from virtual nodes to all real nodes (bidirectional)
            new_edges = []
            # Use BOND_FEAT_DIM + 1 (for virtual indicator) to ensure consistent
            # edge feature dimensions, even for disconnected atoms / single-atom
            # molecules where graph.edge_attr may be empty.
            full_edge_dim = BOND_FEAT_DIM + 1
            new_edge_attrs = []

            for v_idx in range(num_nodes, num_nodes + self.params.num_virtual_nodes):
                for real_idx in range(num_nodes):
                    # Virtual node -> real node
                    new_edges.append([v_idx, real_idx])
                    # Real node -> virtual node
                    new_edges.append([real_idx, v_idx])
                    # Add zero edge features with virtual indicator (1 = virtual)
                    virtual_edge_feat = torch.zeros(full_edge_dim)
                    virtual_edge_feat[-1] = 1.0  # virtual indicator
                    new_edge_attrs.append(virtual_edge_feat)
                    new_edge_attrs.append(virtual_edge_feat.clone())

            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
                graph.edge_index = torch.cat([graph.edge_index, new_edge_tensor], dim=1)
                if new_edge_attrs:
                    new_edge_attr_tensor = torch.stack(new_edge_attrs)
                    # Reshape empty edge_attr to match expected dimension for
                    # disconnected atoms / single-atom molecules with no bonds
                    if graph.edge_attr.numel() == 0:
                        graph.edge_attr = graph.edge_attr.reshape(0, full_edge_dim)
                    graph.edge_attr = torch.cat(
                        [graph.edge_attr, new_edge_attr_tensor], dim=0
                    )

            # Update num_nodes to include virtual nodes
            graph.num_nodes = num_nodes + self.params.num_virtual_nodes

            # Pad positional encodings with zeros for virtual nodes
            if self.params.laplacian_k > 0:
                pad = torch.zeros(
                    (self.params.num_virtual_nodes, self.params.laplacian_k),
                    dtype=torch.float32,
                )
                graph.laplacian_k = torch.cat([graph.laplacian_k, pad], dim=0)

            if self.params.rwse_k > 0:
                pad = torch.zeros(
                    (self.params.num_virtual_nodes, self.params.rwse_k),
                    dtype=torch.float32,
                )
                graph.rwse_k = torch.cat([graph.rwse_k, pad], dim=0)

            if self.params.elstatic_k > 0:
                pad = torch.zeros(
                    (self.params.num_virtual_nodes, self.params.elstatic_k),
                    dtype=torch.float32,
                )
                graph.elstatic_k = torch.cat([graph.elstatic_k, pad], dim=0)

            if self.params.distmat_k > 0:
                pad = torch.zeros(
                    (self.params.num_virtual_nodes, self.params.distmat_k),
                    dtype=torch.float32,
                )
                graph.distmat_k = torch.cat([graph.distmat_k, pad], dim=0)

            if self.params.rrwp_k > 0:
                # RRWP is an edge-level encoding with shape (num_edges, k)
                # We need to pad for the new virtual node edges
                num_virtual_edges = 2 * num_nodes * self.params.num_virtual_nodes
                pad = torch.zeros(
                    (num_virtual_edges, self.params.rrwp_k), dtype=torch.float32
                )
                graph.rrwp_k = torch.cat([graph.rrwp_k, pad], dim=0)

            if self.params.compute_distances and graph.spd is not None:
                # SPD is a 2D tensor (num_nodes, num_nodes) - already handled
                # in _compute_shortest_paths, but we need to recompute with virtual nodes
                # The SPD was computed on the real nodes only, now pad for virtual nodes
                # Use -1 for unreachable (consistent with batch padding and _compute_shortest_paths)
                spd = graph.spd
                total_nodes = num_nodes + self.params.num_virtual_nodes
                full_spd = -torch.ones((total_nodes, total_nodes), dtype=torch.int64)
                full_spd.fill_diagonal_(0)
                full_spd[:num_nodes, :num_nodes] = spd
                graph.spd = full_spd

        return graph

    def _compute_shortest_paths(self, graph: Data, num_real_nodes: int) -> torch.Tensor:
        """Compute shortest path distances for the graph.

        This is called before virtual nodes are added, so we only compute
        SPD for the real molecular graph. Padding for virtual nodes is
        handled in _calculate_graph after virtual nodes are added.

        :param graph: PyG Data object (without virtual nodes)
        :param num_real_nodes: number of real (non-virtual) nodes
        :return: tensor of shortest path distances, with -1 for unreachable pairs
        """

        adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=num_real_nodes)
        adj_csr = adj.tocsr()  # Convert COO to CSR format for shortest_path
        spd = shortest_path(adj_csr, directed=False, unweighted=True)
        # Replace np.inf (unreachable) with -1 before converting to int
        spd = np.where(np.isinf(spd), -1, spd)
        return torch.tensor(spd, dtype=torch.int64)

    def _process_batch(self, mol_batch: list[Mol]) -> list[Data]:
        """Process a batch of molecules in a single process"""
        return [self._calculate_graph(mol, True) for mol in mol_batch]

    def generate_features(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generates unscaled features for graph neural networks, with optional augmentation."""
        # validate inputs without scaling
        mol_list, y, bound_mask, n_jobs = self._validate_input(
            mol_list, y, bound_mask, n_jobs
        )

        graphs = parallelize(self._process_batch, mol_list, n_jobs)

        y_tensor = torch.tensor(y, dtype=torch.float32)
        return StackDataset(graph=graphs, y=y_tensor)

    def featurize(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generates a dataset ready for GNN training.

        Processes a list of N molecules and a numpy array (N,X) encoding the
        labels into a list, which can then be further processed for
        graph neural network training.

        This method chains the unscaled feature generation with
        the necessary Y scaling operations.

        Example usage:

        .. code-block:: python
            GF = GraphDataModulePyG()
            train_dataset = GF.featurize(train_mols, train_y, is_training=True)
            test_dataset = GF.featurize(test_mols, test_y, is_training=False)

        :param list[Mol] mol_list: list of N rdkit molecules to process

        :param np.ndarray | None y: array (N, X), where X is the number of classes or
            endpoints

        :param list[str] | None bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None

        :param bool is_training: Whether to fit Y scaler on the input,
            or leverage a pre-existing one to only normalize

        :param int | None n_jobs: number of cores to use when featurizing the input,
            if None is passed a reasonable n_jobs value will be guessed from the
            amount of data, defaults to None

        :return StackDataset: Processed dataset providing keys 'graph' and 'y' ready
            for graph neural network training
        """
        # Apply augmentation if enabled
        if is_training and self._augment_resonance:
            mol_list, y, bound_mask = self.augment(
                mol_list,
                y,
                bound_mask=bound_mask,
                use_resonance=self._augment_resonance,
                n_jobs=n_jobs,
            )

        # Generate unscaled features
        dataset = self.generate_features(mol_list, y, bound_mask, n_jobs)

        # Apply scaling based on is_training flag
        if is_training:
            self.fit(dataset)

        self.transform(dataset)

        # Handle bound mask and classification transformations (only if not regression)
        self._process_y(dataset, bound_mask)

        return dataset

    def state_dict(self) -> dict:
        """Utility for MLFlow logging"""
        return {
            "ID": "graph",
            "params": self.params.model_dump(),
            "y_scaler": self._y_scaler,
            "label_encoder": self._label_encoder,
            "label_transform": self._label_transform,
            "augment_resonance": self._augment_resonance,
        }

    def load_state_dict(self, state_dict: dict):
        """Utility for MLFlow logging"""
        super().load_state_dict(state_dict)
        self.params = GraphDataModuleInputModel(**state_dict["params"])

    @classmethod
    def dummy(cls):
        """Utility to make a dummy class with default params. Can be
        combined with load_state_dict to recreate a datamodule from
        a state dict
        """
        return cls()


@DataModuleRegistry.register("graph3d")
class Graph3DDataModule(GraphDataModule):
    """3D Graph molecular representation featurization class using PyTorch Geometric.

    Allows users to convert a list of rdkit molecules and labels into a format ready
    to be used for 3D graph neural network training. It uses the Chemprop featurizer
    backend to compute molecular graphs and converts them to PyTorch Geometric format.
    Common featurization logic is inherited from :class:`BaseDataModule`.

    The main purpose of the class is to enable the use of :method:`featurize`.
    Please check out :method:`featurize` for further information on the class' usage.
    """

    def __init__(
        self,
        laplacian_k: int = 10,
        rwse_k: int = 20,
        elstatic_k: int = 0,
        distmat_k: int = 0,
        rrwp_k: int = 20,
        compute_distances: bool = True,
        num_virtual_nodes: int = 0,
        init_virtual_nodes: bool = False,
        embed_timeout: float = 30.0,
        is_classification: bool = False,
        scaler_type: str = "standard",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_params: dict = {},
        batch_size: int = 256,
        num_workers: int = 0,
        augment_resonance: bool = False,
    ):
        super().__init__(
            laplacian_k=laplacian_k,
            rwse_k=rwse_k,
            elstatic_k=elstatic_k,
            distmat_k=distmat_k,
            rrwp_k=rrwp_k,
            compute_distances=compute_distances,
            num_virtual_nodes=num_virtual_nodes,
            init_virtual_nodes=init_virtual_nodes,
            is_classification=is_classification,
            scaler_type=scaler_type,
            clip=clip,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )
        super_params = self.params.model_copy()
        self.params = Graph3DDataModuleInputModel(
            laplacian_k=laplacian_k,
            rwse_k=rwse_k,
            elstatic_k=elstatic_k,
            distmat_k=distmat_k,
            rrwp_k=rrwp_k,
            compute_distances=compute_distances,
            num_virtual_nodes=num_virtual_nodes,
            init_virtual_nodes=init_virtual_nodes,
            embed_timeout=embed_timeout,
            is_classification=is_classification,
            scaler_type=scaler_type,
            clip=clip,
            label_encoder_params=super_params.label_encoder_params,
            label_transform_params=super_params.label_transform_params,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )
        self.collate_fn_map.update({Data: collate_fn_pyg_graph})

    def _calculate_coords(self, mol: Mol) -> torch.Tensor:
        """Computes the 3D atomic coordinates for a given molecule using ETKDG.

        Example usage:

        .. code-block:: python
            coords = get_3D_coords(mol)

        :param rdkit.Chem.rdchem.Mol mol: molecule to compute a conformer for

        :return torch.tensor: tensor (A,3) corresponding to the 3D coordinates
            for each a-th atom in the ETKDG-generated conformer
        """
        # count number of atoms before adding H
        start_n_atoms = mol.GetNumAtoms()

        # add Hs to get reasonable conformers
        mol = Chem.AddHs(mol)

        # Use ETKDGv3 for conformer generation
        etkdg_params = AllChem.ETKDGv3()

        # try to embed molecule with ETKDG
        flag = _embed_with_timeout(mol, etkdg_params, self.params.embed_timeout)

        # if embedding failed, retry with random coordinates as a fallback
        if flag == -1:
            etkdg_params.useRandomCoords = True
            flag = _embed_with_timeout(mol, etkdg_params, self.params.embed_timeout)

        # if all embedding attempts fail, fall back to 2D coordinates
        if flag == -1:
            AllChem.Compute2DCoords(mol)

        # get coords as numpy array of shape (N+n_H, 3)
        try:
            conf = mol.GetConformer()
            coords = conf.GetPositions()
        except Exception:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            coords = conf.GetPositions()

        # slice coord array only on non-H indexes
        non_h_idx = []
        for idx in range(start_n_atoms):
            non_h_idx.append(idx)
        coords = coords[non_h_idx, :]

        # add coords for virtual node
        if self.params.num_virtual_nodes > 0:
            virtual_coords = np.zeros((self.params.num_virtual_nodes, 3))
            coords = np.concatenate((coords, virtual_coords), axis=0)

        return torch.tensor(coords, dtype=torch.float32)

    def _calculate_graph_with_pos(self, mol: Mol, is_training: bool = True) -> Data:
        """Build a PyG :class:`Data` for ``mol`` and attach 3D coords to ``pos``.

        Coordinates ride on ``Data.pos`` (the PyG convention) so that
        :class:`torch_geometric.data.Batch` auto-concatenates them alongside
        ``x`` and the per-node positional encodings — no parallel ``coords``
        collate key is needed, and all 3D encoders read the coords from
        ``graph.pos`` inside their per-layer hook.

        :param Mol mol: molecule to convert.
        :param bool is_training: forwarded to :meth:`_calculate_graph`
            (currently unused, kept for signature parity).
        :returns: PyG :class:`Data` with 3D coordinates attached to ``pos``.
        :rtype: Data
        """
        graph = self._calculate_graph(mol, is_training)
        graph.pos = self._calculate_coords(mol)
        return graph

    def _process_batch(self, mol_batch: list[Mol]) -> list[Data]:
        """Process a batch of molecules in a single process, attaching coords."""
        return [self._calculate_graph_with_pos(mol, True) for mol in mol_batch]

    def generate_features(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generates unscaled features for 3D graph neural networks, with optional augmentation."""
        # validate inputs without scaling
        mol_list, y, bound_mask, n_jobs = self._validate_input(
            mol_list, y, bound_mask, n_jobs
        )

        # Apply augmentation if enabled
        if self._augment_resonance:
            mol_list, y, bound_mask = self.augment(
                mol_list,
                y,
                bound_mask=bound_mask,
                use_resonance=self._augment_resonance,
                n_jobs=n_jobs,
            )

        graphs = parallelize(self._process_batch, mol_list, n_jobs)

        y_tensor = torch.tensor(y, dtype=torch.float32)
        return StackDataset(graph=graphs, y=y_tensor)

    def state_dict(self) -> dict:
        """Utility for MLFlow logging"""
        return {
            "ID": "graph3d",
            "params": self.params.model_dump(),
            "y_scaler": self._y_scaler,
            "label_encoder": self._label_encoder,
            "label_transform": self._label_transform,
            "augment_resonance": self._augment_resonance,
        }

    def load_state_dict(self, state_dict: dict):
        """Utility for MLFlow logging"""
        BaseDataModule.load_state_dict(self, state_dict)
        self.params = Graph3DDataModuleInputModel(**state_dict["params"])
