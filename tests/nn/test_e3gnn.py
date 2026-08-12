"""Unit tests for EGNN_Sparse and E3GNN encoder."""

import torch
from torch_geometric.data import Batch, Data

from matcha.torch.encoders.e3gnn import E3GNN, EGNN_Sparse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOM_INPUT_DIM = 8
_ATOM_HIDDEN_DIM = 16
_M_DIM = 8


def _make_encoder(dropout: float = 0.0) -> E3GNN:
    return E3GNN(
        num_layers=1,
        atom_input_dim=_ATOM_INPUT_DIM,
        bond_input_dim=0,
        atom_hidden_dim=_ATOM_HIDDEN_DIM,
        m_dim=_M_DIM,
        fourier_features=0,
        soft_edge=False,
        norm_feats=False,
        norm_coors=False,
        update_coors=True,
        activation="relu",
        dropout=dropout,
        coor_weights_clamp_value=100.0,
        norm_coors_scale_init=1e-2,
        jk="last",
        readout="mean",
        laplacian_k=0,
        rwse_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
    )


def _make_batch(n_atoms: int = 4) -> tuple[Batch, torch.Tensor]:
    """Build a single-molecule PyG batch and separate coordinate tensor."""
    edge_index = torch.stack(
        [
            torch.arange(n_atoms - 1),
            torch.arange(1, n_atoms),
        ],
        dim=0,
    )
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    g = Data(x=torch.randn(n_atoms, _ATOM_INPUT_DIM), edge_index=edge_index)
    batch = Batch.from_data_list([g])
    coords = torch.randn(n_atoms, 3)
    return batch, coords


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rotation_invariance():
    """Graph-level output is invariant to rigid rotations of atom coordinates."""
    encoder = _make_encoder()
    encoder.eval()

    batch, coords = _make_batch()

    with torch.no_grad():
        out_orig = encoder(batch, coords)

        # Random SO(3) rotation via QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if Q.det() < 0:
            Q[:, 0] = -Q[:, 0]
        coords_rot = (Q @ coords.T).T

        out_rot = encoder(batch, coords_rot)

    assert out_orig.shape == out_rot.shape
    assert torch.allclose(out_orig, out_rot, atol=1e-5), (
        f"Outputs differ after rotation: max diff = {(out_orig - out_rot).abs().max():.2e}"
    )


def test_eval_mode_determinism():
    """Two forward passes in eval mode with dropout>0 produce identical results."""
    encoder = _make_encoder(dropout=0.5)
    encoder.eval()

    batch, coords = _make_batch()

    with torch.no_grad():
        out1 = encoder(batch, coords)
        out2 = encoder(batch, coords)

    assert torch.equal(out1, out2), "Eval-mode outputs are not bit-identical"


def test_coord_update_magnitude_bounded():
    """Coordinate updates stay small with paper defaults (clamp + mean + small init).

    Regression net for the Stage 1 bundle: clamp=100 + mean coord aggregation +
    coors_mlp last-layer init with gain=1e-3. Any of these silently reverting
    (e.g. clamp=None, aggr='add' on a high-degree graph, default-Xavier init)
    lets per-atom coord deltas blow up on a dense graph.
    """
    torch.manual_seed(0)
    layer = EGNN_Sparse(
        feats_dim=_ATOM_HIDDEN_DIM,
        edge_attr_dim=0,
        m_dim=_M_DIM,
        fourier_features=0,
        soft_edge=False,
        norm_feats=False,
        norm_coors=False,
        update_feats=True,
        update_coors=True,
        dropout=0.0,
        coor_weights_clamp_value=100.0,
        aggr="add",
        coord_aggr="mean",
    )
    layer.eval()

    # High per-node degree: fully connected 6-atom graph, both directions.
    n_atoms = 6
    edge_index = torch.combinations(torch.arange(n_atoms), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    coords = torch.randn(n_atoms, 3)
    feats = torch.randn(n_atoms, _ATOM_HIDDEN_DIM)

    with torch.no_grad():
        coords_out, _ = layer(coords, feats, edge_index)

    delta = (coords_out - coords).norm(dim=-1)
    assert torch.isfinite(delta).all(), "Coord update is non-finite"
    # gain=1e-3 init keeps the initial update near-identity; anything above
    # this ceiling means the near-identity guarantee has regressed.
    assert delta.max().item() < 1.0, (
        f"Max per-atom coord delta {delta.max().item():.3e} exceeds expected bound"
    )
