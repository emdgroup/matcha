"""
Graph Positional Encodings for PyTorch Geometric graphs.

This module provides various positional encoding methods for molecular graphs
represented as PyTorch Geometric Data objects.
"""

import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, GetDistanceMatrix
from scipy.linalg import pinv
import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj


class GraphPE:
    """Class providing static methods for computing various positional encodings
    on PyTorch Geometric graphs.
    """

    @classmethod
    def _get_adjacency_matrix(cls, graph: Data) -> torch.Tensor:
        """Get dense adjacency matrix from a PyG graph.

        :param graph: PyTorch Geometric Data object
        :return: Dense adjacency matrix (N, N)
        """
        num_nodes = graph.num_nodes
        adj = to_dense_adj(graph.edge_index, max_num_nodes=num_nodes)
        return adj.squeeze(0)  # Remove batch dimension

    @classmethod
    def _handle_pe_exception(cls, num_nodes: int, k: int) -> torch.Tensor:
        """Return zero tensor when PE computation fails.

        :param num_nodes: number of nodes in the graph
        :param k: dimensionality of the positional encoding
        :return: zero tensor of shape (num_nodes, k)
        """
        return torch.zeros((num_nodes, k), dtype=torch.float32)

    @classmethod
    def lap_pe(cls, graph: Data, k: int) -> torch.Tensor:
        """Compute Laplacian Positional Encoding.

        Uses the eigenvectors of the graph Laplacian as positional encodings.

        :param graph: PyTorch Geometric Data object
        :param k: number of eigenvectors to use
        :return: tensor of shape (num_nodes, k) containing Laplacian PE
        """
        try:
            num_nodes = graph.num_nodes

            # Get Laplacian matrix
            edge_index, edge_weight = get_laplacian(
                graph.edge_index, normalization="sym", num_nodes=num_nodes
            )

            # Convert to dense matrix
            L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)
            L = L.squeeze(0)  # Remove batch dimension

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(L)

            # Sort by eigenvalues (ascending)
            idx = torch.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx]

            # Take the first k+1 eigenvectors (skip the first constant eigenvector)
            # and take absolute values (to handle sign ambiguity)
            if num_nodes <= k:
                # Pad with zeros if not enough nodes
                pe = torch.zeros((num_nodes, k), dtype=torch.float32)
                pe[:, : min(num_nodes - 1, k)] = torch.abs(
                    eigenvectors[:, 1 : min(num_nodes, k + 1)]
                )
            else:
                pe = torch.abs(eigenvectors[:, 1 : k + 1])

            return pe.float()

        except Exception:
            return cls._handle_pe_exception(graph.num_nodes, k)

    @classmethod
    def rw_se(cls, graph: Data, k: int) -> torch.Tensor:
        """Compute Random Walk Structural Encoding.

        Uses the diagonal of powers of the random walk transition matrix.

        :param graph: PyTorch Geometric Data object
        :param k: number of random walk steps
        :return: tensor of shape (num_nodes, k) containing RWSE
        """
        try:
            num_nodes = graph.num_nodes

            # Get adjacency matrix
            adj = cls._get_adjacency_matrix(graph)

            # Compute degree matrix
            degrees = adj.sum(dim=1)
            degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)

            # Random walk transition matrix: D^{-1} * A
            D_inv = torch.diag(1.0 / degrees)
            M = torch.mm(D_inv, adj)

            # Compute diagonal of M^1, M^2, ..., M^k
            rwse = torch.zeros((num_nodes, k), dtype=torch.float32)
            M_power = M.clone()

            for step in range(k):
                rwse[:, step] = torch.diag(M_power)
                M_power = torch.mm(M_power, M)

            return rwse

        except Exception:
            return cls._handle_pe_exception(graph.num_nodes, k)

    @classmethod
    def rrwp_re(cls, graph: Data, k: int) -> torch.Tensor:
        """Compute Relative Random Walk Probabilities for edges.

        For each edge (i, j), computes the k-step random walk probabilities
        from i to j.

        :param graph: PyTorch Geometric Data object
        :param k: number of random walk steps
        :return: tensor of shape (num_edges, k) containing RRWP
        """
        try:
            num_nodes = graph.num_nodes

            # Get adjacency matrix
            adj = cls._get_adjacency_matrix(graph)

            # Compute degree matrix
            degrees = adj.sum(dim=1)
            degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)

            # Random walk transition matrix: D^{-1} * A
            D_inv = torch.diag(1.0 / degrees)
            M = torch.mm(D_inv, adj)

            # Compute P tensor: P[:, :, step] = M^step
            P = torch.zeros(num_nodes, num_nodes, k, dtype=torch.float32)
            P[:, :, 0] = torch.eye(num_nodes)

            if k > 1:
                P[:, :, 1] = M

            M_power = M.clone()
            for step in range(2, k):
                M_power = torch.mm(M_power, M)
                P[:, :, step] = M_power

            # Extract edge encodings
            src, dst = graph.edge_index
            edge_encodings = P[src, dst, :]

            return edge_encodings

        except Exception:
            num_edges = graph.edge_index.shape[1]
            return torch.zeros((num_edges, k), dtype=torch.float32)

    @classmethod
    def elstatic_pe(cls, mol: Mol, k: int) -> torch.Tensor:
        """Compute electrostatic-inspired positional encoding.

        Uses statistics of the pseudo-inverse of the graph Laplacian
        derived from the adjacency matrix.

        :param mol: RDKit molecule object
        :param k: number of statistical features to compute
        :return: tensor of shape (num_atoms, k) containing electrostatic PE
        """
        return cls._pinv_embedding(mol, k, "adj")

    @classmethod
    def distmat_pe(cls, mol: Mol, k: int) -> torch.Tensor:
        """Compute distance matrix-based positional encoding.

        Uses statistics of the pseudo-inverse of the graph Laplacian
        derived from the distance matrix.

        :param mol: RDKit molecule object
        :param k: number of statistical features to compute
        :return: tensor of shape (num_atoms, k) containing distance matrix PE
        """
        return cls._pinv_embedding(mol, k, "dist")

    @classmethod
    def _pinv_embedding(cls, mol: Mol, k: int, fn: str) -> torch.Tensor:
        """Compute pseudo-inverse based positional encoding.

        :param mol: RDKit molecule object
        :param k: number of statistical features
        :param fn: 'adj' for adjacency matrix, 'dist' for distance matrix
        :return: tensor of shape (num_atoms, k)
        """
        stats = [
            np.min,
            np.max,
            np.mean,
            np.median,
            np.std,
            lambda x, axis=1: np.sum(x**2, axis=axis),
            lambda x, axis=1: np.sum(np.abs(x), axis=axis),
            lambda x, axis=1: np.mean(x**3, axis=axis),
            lambda x, axis=1: np.mean(x**4, axis=axis),
            lambda x, axis=1: np.sum(
                x > np.mean(x, axis=axis, keepdims=True), axis=axis
            ),
            lambda x, axis=1: np.sum(x > 0, axis=axis),
            lambda x, axis=1: np.sum(x < 0, axis=axis),
        ]

        if k > len(stats):
            raise ValueError(f"PE embedding k must be less or equal than {len(stats)}")

        stat_list = stats[:k]

        if fn == "adj":
            adj = GetAdjacencyMatrix(mol)
        elif fn == "dist":
            adj = GetDistanceMatrix(mol, useBO=True, useAtomWts=True)
        else:
            raise ValueError(f"Unknown function type: {fn}")

        L = np.diag(np.sum(adj, axis=1)) - adj
        L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
        inv = pinv(L)
        elstatic = inv - np.diag(np.diag(inv))

        embedding = []
        for stat in stat_list:
            result = stat(elstatic, axis=1)
            if np.isscalar(result) or result.ndim == 0:
                result = np.full(elstatic.shape[0], result)
            embedding.append(result)

        pe = np.stack(embedding, axis=-1)

        return torch.tensor(pe, dtype=torch.float32)
