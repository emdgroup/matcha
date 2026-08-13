"""Tests for :class:`matcha.torch.encoders.attentivefp.AttentiveFP`.

Covers the base-encoder contract introduced by issue #24:
:meth:`forward_nodes_per_layer` produces the per-layer node embeddings
consumed by both the classic and pretraining paths, and
:meth:`BaseGraphEncoder.forward` is a thin template that combines those
embeddings via jumping knowledge and the configured readout.
"""

import pytest
import torch


pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402

from matcha.torch.encoders.attentivefp import AttentiveFP  # noqa: E402


_NUM_LAYERS = 3
_ATOM_INPUT_DIM = 8
_BOND_INPUT_DIM = 4
_ATOM_HIDDEN_DIM = 16


def _make_attentivefp(**overrides) -> AttentiveFP:
    """Build a tiny AttentiveFP encoder with all positional encodings disabled."""
    kwargs = dict(
        num_layers=_NUM_LAYERS,
        atom_input_dim=_ATOM_INPUT_DIM,
        bond_input_dim=_BOND_INPUT_DIM,
        atom_hidden_dim=_ATOM_HIDDEN_DIM,
        dropout=0.0,
        readout="sum",
        jk="last",
        laplacian_k=0,
        rwse_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
    )
    kwargs.update(overrides)
    return AttentiveFP(**kwargs)


def _make_batch(batch_size: int = 2) -> Batch:
    """Small PyG batch of chain graphs with random node/edge features."""
    graphs = []
    for _ in range(batch_size):
        n_nodes = 4
        src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
        dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        graphs.append(
            Data(
                x=torch.randn(n_nodes, _ATOM_INPUT_DIM),
                edge_index=edge_index,
                edge_attr=torch.randn(edge_index.size(1), _BOND_INPUT_DIM),
            )
        )
    return Batch.from_data_list(graphs)


def test_forward_nodes_per_layer_returns_one_tensor_per_layer():
    """AttentiveFP builds one initial context layer plus ``num_layers - 1``
    subsequent GATv2Conv+GRU layers, so the returned list still has length
    ``num_layers``."""
    torch.manual_seed(0)
    encoder = _make_attentivefp()
    encoder.eval()

    with torch.no_grad():
        per_layer, g = encoder.forward_nodes_per_layer(_make_batch())

    assert isinstance(per_layer, list)
    assert len(per_layer) == _NUM_LAYERS
    total_nodes = int(g.batch.numel())
    for feats in per_layer:
        assert feats.shape == (total_nodes, _ATOM_HIDDEN_DIM)


def test_forward_uses_template_from_base_graph_encoder():
    """``forward`` must equal ``readout(g, _run_jk(forward_nodes_per_layer))``.

    The template method lives on :class:`BaseGraphEncoder`; this test
    verifies that :class:`AttentiveFP` inherits it (i.e. did not shadow it)
    and that the layer loop still drives the same output the classic path
    sees.
    """
    torch.manual_seed(0)
    encoder = _make_attentivefp(jk="concat", readout="sum")
    encoder.eval()

    batch = _make_batch()

    with torch.no_grad():
        per_layer, g = encoder.forward_nodes_per_layer(batch)
        merged = encoder._run_jk(per_layer)
        expected = encoder.readout(g, merged)
        actual = encoder(batch)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual.shape == (2, _ATOM_HIDDEN_DIM * _NUM_LAYERS)


@pytest.mark.parametrize("jk", ["last", "concat", "sum", "max"])
def test_forward_output_shape_matches_jk(jk: str):
    torch.manual_seed(0)
    encoder = _make_attentivefp(jk=jk, readout="sum")
    encoder.eval()

    with torch.no_grad():
        out = encoder(_make_batch(batch_size=2))

    expected_dim = (
        _ATOM_HIDDEN_DIM * _NUM_LAYERS if jk == "concat" else _ATOM_HIDDEN_DIM
    )
    assert out.shape == (2, expected_dim)
    assert encoder.fp_dim == expected_dim
