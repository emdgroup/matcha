"""Unit tests for MPNNPlusConv and GPSBlock residual connections."""

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from matcha.nn.layers import BiasedMultiHeadAttention
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


def test_mpnn_node_pure_delta():
    """With zeroed node_model weights, MPNNPlusConv node delta is zero.

    MPNNPlusConv returns pure deltas (no internal residual); the outer
    GPSBlock provides the single symmetric skip over both MP and attention
    branches.
    """
    mpnn = _make_mpnn()
    for p in mpnn.node_model.parameters():
        nn.init.zeros_(p)

    n_nodes = _N_NODES
    edge_index = _chain_edge_index(n_nodes)
    x = torch.randn(n_nodes, _ATOM_FEATS)
    edge_attr = torch.randn(edge_index.size(1), _EDGE_FEATS)

    with torch.no_grad():
        x_delta, _ = mpnn(x, edge_index, edge_attr)

    assert torch.allclose(x_delta, torch.zeros_like(x), atol=1e-6), (
        f"Node delta should be zero when node_model is zeroed; "
        f"max abs = {x_delta.abs().max():.2e}"
    )


def test_mpnn_edge_pure_delta():
    """With zeroed edge_model weights, MPNNPlusConv edge delta is zero."""
    mpnn = _make_mpnn()
    for p in mpnn.edge_model.parameters():
        nn.init.zeros_(p)

    n_nodes = _N_NODES
    edge_index = _chain_edge_index(n_nodes)
    x = torch.randn(n_nodes, _ATOM_FEATS)
    edge_attr = torch.randn(edge_index.size(1), _EDGE_FEATS)

    with torch.no_grad():
        _, edge_delta = mpnn(x, edge_index, edge_attr)

    assert torch.allclose(edge_delta, torch.zeros_like(edge_attr), atol=1e-6), (
        f"Edge delta should be zero when edge_model is zeroed; "
        f"max abs = {edge_delta.abs().max():.2e}"
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


def test_biased_mha_fully_padded_query_row_no_nan_grads():
    """A fully-padded query row must not produce NaN gradients through WV/WO.

    When ``to_dense_batch`` pads uneven batches, some query rows are entirely
    masked. The naive softmax(−inf, ..., −inf) → NaN would propagate through
    the value projection and the output projection during backward. The
    safe-softmax path in ``BiasedMultiHeadAttention`` replaces those rows with
    a finite constant, so no NaN gradients should appear.
    """
    torch.manual_seed(0)
    embed_dim = 16
    num_heads = 4
    batch_size = 2
    seq_len = 5

    mha = BiasedMultiHeadAttention(embed_dim, num_heads, dropout=0.0)
    mha.train()

    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    # Second graph is fully padded — all query rows are invalid.
    attn_mask = torch.tensor(
        [
            [True, True, True, False, False],  # graph 0: 3 real nodes
            [False, False, False, False, False],  # graph 1: all padded
        ]
    )

    out = mha(x, attn_bias=None, attn_mask=attn_mask)
    loss = out.sum()
    loss.backward()

    for name, p in mha.named_parameters():
        assert p.grad is not None, f"No gradient reached {name}"
        assert not torch.isnan(p.grad).any(), (
            f"NaN gradient in {name} — safe-softmax did not protect fully-"
            f"padded query rows."
        )
    assert x.grad is not None, "No gradient reached input x"
    assert not torch.isnan(x.grad).any(), "NaN gradient in input x"


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
