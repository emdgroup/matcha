"""Tests for GraphPE (graph positional encodings)."""

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

from matcha.datamodules.classic.graph_positional_encoder import GraphPE


@pytest.fixture
def simple_graph():
    """A linear graph: 0 -- 1 -- 2 -- 3."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    x = torch.ones(4, 1)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


@pytest.fixture
def benzene_mol():
    """RDKit molecule for benzene."""
    return Chem.MolFromSmiles("c1ccccc1")


@pytest.fixture
def benzene_graph():
    """PyG graph approximating benzene (cycle of 6)."""
    edges = []
    for i in range(6):
        edges.append([i, (i + 1) % 6])
        edges.append([(i + 1) % 6, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.ones(6, 1)
    return Data(x=x, edge_index=edge_index, num_nodes=6)


# ===================================================================
# Laplacian PE
# ===================================================================


class TestLaplacianPE:
    def test_shape(self, simple_graph):
        pe = GraphPE.lap_pe(simple_graph, k=3)
        assert pe.shape == (4, 3)

    def test_dtype(self, simple_graph):
        pe = GraphPE.lap_pe(simple_graph, k=3)
        assert pe.dtype == torch.float32

    def test_non_negative(self, simple_graph):
        pe = GraphPE.lap_pe(simple_graph, k=3)
        # Absolute values are taken, so all should be >= 0
        assert (pe >= 0).all()

    def test_padding_when_k_exceeds_nodes(self):
        """When k > num_nodes - 1, extra columns should be zero."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g = Data(x=torch.ones(2, 1), edge_index=edge_index, num_nodes=2)
        pe = GraphPE.lap_pe(g, k=5)
        assert pe.shape == (2, 5)
        # Only 1 non-trivial eigenvector for 2 nodes
        assert (pe[:, 1:] == 0).all()


# ===================================================================
# Random Walk SE
# ===================================================================


class TestRWSE:
    def test_shape(self, simple_graph):
        pe = GraphPE.rw_se(simple_graph, k=5)
        assert pe.shape == (4, 5)

    def test_dtype(self, simple_graph):
        pe = GraphPE.rw_se(simple_graph, k=5)
        assert pe.dtype == torch.float32

    def test_first_step_diagonal(self, simple_graph):
        """The first random walk step diagonal should be the self-return probability."""
        pe = GraphPE.rw_se(simple_graph, k=3)
        # For a linear graph, nodes 0 and 3 have degree 1, so self-return after 1 step = 0
        # (they can only go to one neighbor)
        assert pe[0, 0].item() == 0.0


# ===================================================================
# RRWP (Relative Random Walk Probabilities)
# ===================================================================


class TestRRWP:
    def test_shape(self, simple_graph):
        pe = GraphPE.rrwp_re(simple_graph, k=4)
        num_edges = simple_graph.edge_index.shape[1]
        assert pe.shape == (num_edges, 4)

    def test_dtype(self, simple_graph):
        pe = GraphPE.rrwp_re(simple_graph, k=4)
        assert pe.dtype == torch.float32

    def test_first_step_identity(self, simple_graph):
        """Step 0 is the identity – should be 0 for all edges (i ≠ j)."""
        pe = GraphPE.rrwp_re(simple_graph, k=3)
        # Column 0 = P^0[src, dst] = I[src, dst], all non-self edges => 0
        assert (pe[:, 0] == 0).all()


# ===================================================================
# Electrostatic PE
# ===================================================================


class TestElstaticPE:
    def test_shape(self, benzene_mol):
        pe = GraphPE.elstatic_pe(benzene_mol, k=5)
        num_atoms = benzene_mol.GetNumAtoms()
        assert pe.shape == (num_atoms, 5)

    def test_dtype(self, benzene_mol):
        pe = GraphPE.elstatic_pe(benzene_mol, k=5)
        assert pe.dtype == torch.float32

    def test_k_too_large_raises(self, benzene_mol):
        with pytest.raises(ValueError):
            GraphPE.elstatic_pe(benzene_mol, k=100)


# ===================================================================
# Distance Matrix PE
# ===================================================================


class TestDistmatPE:
    def test_shape(self, benzene_mol):
        pe = GraphPE.distmat_pe(benzene_mol, k=5)
        num_atoms = benzene_mol.GetNumAtoms()
        assert pe.shape == (num_atoms, 5)

    def test_dtype(self, benzene_mol):
        pe = GraphPE.distmat_pe(benzene_mol, k=5)
        assert pe.dtype == torch.float32


# ===================================================================
# Helper – _handle_pe_exception
# ===================================================================


class TestHandlePEException:
    def test_returns_zeros(self):
        result = GraphPE._handle_pe_exception(10, 5)
        assert result.shape == (10, 5)
        assert (result == 0).all()
