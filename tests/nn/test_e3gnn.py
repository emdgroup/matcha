"""Unit tests for EGNN_Sparse and E3GNN encoder."""

import torch
from torch_geometric.data import Batch, Data

from matcha.torch.encoders.e3gnn import E3GNN

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
