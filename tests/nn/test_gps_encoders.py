"""Unit tests for MPNNPlusConv and GPSBlock residual connections."""

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from matcha.torch.encoders.gps import GPS, GPSBlock, MPNNPlusConv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ATOM_FEATS = 16
_EDGE_FEATS = 16
_NUM_HEADS = 4
_N_NODES = 6
_N_EDGES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mpnn(
    in_dim: int = _ATOM_FEATS,
    out_dim: int = _ATOM_FEATS,
    in_dim_edges: int = _EDGE_FEATS,
    out_dim_edges: int = _EDGE_FEATS,
) -> MPNNPlusConv:
    return MPNNPlusConv(
        in_dim=in_dim,
        out_dim=out_dim,
        in_dim_edges=in_dim_edges,
        out_dim_edges=out_dim_edges,
        activation="relu",
        dropout=0.0,
        edge_dropout=0.0,
        mlp_expansion_ratio=2,
    )


def _make_gpsblock(
    atom_feats: int = _ATOM_FEATS,
    edge_feats: int = _EDGE_FEATS,
    num_heads: int = _NUM_HEADS,
) -> GPSBlock:
    return GPSBlock(
        atom_feats=atom_feats,
        edge_feats=edge_feats,
        dropout=0.0,
        norm="adarmsn",
        activation="relu",
        num_heads=num_heads,
        expansion_k=2,
    )


def _make_gps_encoder(
    atom_input_dim: int = _ATOM_FEATS,
    bond_input_dim: int = _EDGE_FEATS,
    atom_hidden_dim: int = _ATOM_FEATS,
    num_heads: int = _NUM_HEADS,
    num_layers: int = 2,
) -> GPS:
    return GPS(
        num_layers=num_layers,
        atom_input_dim=atom_input_dim,
        bond_input_dim=bond_input_dim,
        atom_hidden_dim=atom_hidden_dim,
        num_heads=num_heads,
        expansion_k=2,
        distance_k=None,
        activation="relu",
        dropout=0.0,
        norm="adarmsn",
        jk="last",
        readout="mean",
        laplacian_k=0,
        rwse_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
    )


def _make_graph_data(
    n_nodes: int = _N_NODES,
    atom_input_dim: int = _ATOM_FEATS,
    bond_input_dim: int = _EDGE_FEATS,
) -> Batch:
    """Create a minimal single-molecule PyG batch for GPS forward tests."""
    # Chain graph: 0-1-2-...(n_nodes-1), undirected
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    n_edges = edge_index.size(1)
    g = Data(
        x=torch.randn(n_nodes, atom_input_dim),
        edge_index=edge_index,
        edge_attr=torch.randn(n_edges, bond_input_dim),
    )
    return Batch.from_data_list([g])


def _chain_edge_index(n: int) -> torch.Tensor:
    """Undirected chain 0-1-...(n-1)."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mpnn_node_residual():
    """With zeroed node_model weights, MPNNPlusConv node output equals input."""
    mpnn = _make_mpnn()
    for p in mpnn.node_model.parameters():
        nn.init.zeros_(p)

    n_nodes = _N_NODES
    edge_index = _chain_edge_index(n_nodes)
    x = torch.randn(n_nodes, _ATOM_FEATS)
    edge_attr = torch.randn(edge_index.size(1), _EDGE_FEATS)

    with torch.no_grad():
        x_out, _ = mpnn(x, edge_index, edge_attr)

    assert torch.allclose(x_out, x, atol=1e-6), (
        f"Node output should equal input when node_model is zeroed; "
        f"max diff = {(x_out - x).abs().max():.2e}"
    )


def test_mpnn_edge_residual():
    """With zeroed edge_model weights, MPNNPlusConv edge output equals edge input."""
    mpnn = _make_mpnn()
    for p in mpnn.edge_model.parameters():
        nn.init.zeros_(p)

    n_nodes = _N_NODES
    edge_index = _chain_edge_index(n_nodes)
    x = torch.randn(n_nodes, _ATOM_FEATS)
    edge_attr = torch.randn(edge_index.size(1), _EDGE_FEATS)

    with torch.no_grad():
        _, edge_out = mpnn(x, edge_index, edge_attr)

    assert torch.allclose(edge_out, edge_attr, atol=1e-6), (
        f"Edge output should equal input when edge_model is zeroed; "
        f"max diff = {(edge_out - edge_attr).abs().max():.2e}"
    )


def test_gpsblock_h_in_residual():
    """With all weights zeroed, GPSBlock output equals input (identity at zero-init)."""
    block = _make_gpsblock()
    for p in block.parameters():
        nn.init.zeros_(p)
    block.eval()

    n_nodes = _N_NODES
    edge_index = _chain_edge_index(n_nodes)
    graph = Batch.from_data_list(
        [
            Data(
                x=torch.zeros(n_nodes, _ATOM_FEATS),
                edge_index=edge_index,
                edge_attr=torch.zeros(edge_index.size(1), _EDGE_FEATS),
            )
        ]
    )
    feat = torch.randn(n_nodes, _ATOM_FEATS)
    edge_feat = torch.randn(edge_index.size(1), _EDGE_FEATS)
    graph_id = graph.batch

    with torch.no_grad():
        out_feat, _ = block(graph, feat, edge_feat, graph_id, dist_bias=None)

    assert torch.allclose(out_feat, feat, atol=1e-6), (
        f"GPSBlock with zeroed weights should be identity; "
        f"max diff = {(out_feat - feat).abs().max():.2e}"
    )


def test_gpsblock_gradient_flow():
    """Gradients reach the input tensor through the h_in skip connection."""
    block = _make_gpsblock()

    n_nodes = _N_NODES
    edge_index = _chain_edge_index(n_nodes)
    graph = Batch.from_data_list(
        [
            Data(
                x=torch.zeros(n_nodes, _ATOM_FEATS),
                edge_index=edge_index,
                edge_attr=torch.zeros(edge_index.size(1), _EDGE_FEATS),
            )
        ]
    )
    feat = torch.randn(n_nodes, _ATOM_FEATS, requires_grad=True)
    edge_feat = torch.randn(edge_index.size(1), _EDGE_FEATS)
    graph_id = graph.batch

    out_feat, _ = block(graph, feat, edge_feat, graph_id, dist_bias=None)
    out_feat.sum().backward()

    assert feat.grad is not None, "No gradient reached the input feat tensor"
    assert feat.grad.abs().sum() > 0, (
        "Gradient is all-zero — skip connection not contributing"
    )


def test_gps_encoder_forward_shape():
    """GPS encoder produces output of expected shape on a synthetic batch."""
    atom_input_dim = _ATOM_FEATS
    bond_input_dim = _EDGE_FEATS
    atom_hidden_dim = _ATOM_FEATS
    n_nodes = _N_NODES

    encoder = _make_gps_encoder(
        atom_input_dim=atom_input_dim,
        bond_input_dim=bond_input_dim,
        atom_hidden_dim=atom_hidden_dim,
    )
    encoder.eval()

    batch = _make_graph_data(
        n_nodes=n_nodes,
        atom_input_dim=atom_input_dim,
        bond_input_dim=bond_input_dim,
    )

    with torch.no_grad():
        out = encoder(batch)

    # One molecule in batch → shape [1, atom_hidden_dim]
    assert out.shape == (1, atom_hidden_dim), (
        f"Expected GPS output shape (1, {atom_hidden_dim}), got {tuple(out.shape)}"
    )
