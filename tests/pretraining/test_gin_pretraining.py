"""Tests for :class:`matcha.torch.models.pretraining.gin_pretraining.GINPretraining`.

Focus of these tests is the Stage-1 unification from issue #24:
- the pretraining model consumes the canonical :class:`GIN` encoder rather
  than a pretraining-specific twin;
- the ``enc_eps`` / ``enc_train_eps`` hyperparameters flow through
  ``_build_encoder`` into ``GINEConv``;
- an end-to-end forward call still returns the expected ``{"node", "graph"}``
  prediction shapes.
"""

import pytest
import torch


pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.nn import GINEConv  # noqa: E402

from matcha.datamodules.classic.graph_datamodule import (  # noqa: E402
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
)
from matcha.torch.encoders.gin import GIN  # noqa: E402
from matcha.torch.models.pretraining.gin_pretraining import GINPretraining  # noqa: E402


def _make_model(**overrides) -> GINPretraining:
    """Build a tiny GINPretraining with all positional encodings disabled."""
    kwargs = dict(
        num_node_targets=2,
        num_graph_targets=1,
        enc_num_layers=2,
        enc_atom_hidden_dim=16,
        enc_norm=None,
        enc_readout="sum",
        enc_aggregation="sum",
        enc_activation="relu",
        enc_dropout=0.0,
        enc_laplacian_k=0,
        enc_rwse_k=0,
        enc_elstatic_k=0,
        enc_distmat_k=0,
        enc_rrwp_k=0,
        node_head_dims=[8],
        graph_head_dims=[8],
        pred_activation="relu",
        pred_dropout=0.0,
    )
    kwargs.update(overrides)
    return GINPretraining(**kwargs)


def _make_batch(batch_size: int = 2, n_nodes_per_graph: int = 3):
    """Minimal batch dict for :meth:`GINPretraining.forward`."""
    graphs = []
    for _ in range(batch_size):
        src = list(range(n_nodes_per_graph - 1)) + list(range(1, n_nodes_per_graph))
        dst = list(range(1, n_nodes_per_graph)) + list(range(n_nodes_per_graph - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        graphs.append(
            Data(
                x=torch.randn(n_nodes_per_graph, ATOM_FEAT_DIM),
                edge_index=edge_index,
                edge_attr=torch.randn(edge_index.size(1), BOND_FEAT_DIM),
            )
        )
    graph = Batch.from_data_list(graphs)
    return {
        "graph": graph,
        "y_node": torch.randn(batch_size * n_nodes_per_graph, 2),
        "y_graph": torch.randn(batch_size, 1),
    }


def test_encoder_is_canonical_gin():
    """The pretraining model wires up the canonical ``GIN`` encoder."""
    model = _make_model()
    assert isinstance(model.encoder, GIN)


def test_enc_eps_and_train_eps_flow_into_gineconv():
    """``enc_eps`` and ``enc_train_eps`` are forwarded to every ``GINEConv``."""
    model = _make_model(enc_eps=0.25, enc_train_eps=True)

    for layer in model.encoder.layers:
        assert isinstance(layer, GINEConv)
        assert pytest.approx(0.25) == layer.eps.detach().item()
        # train_eps=True means eps is a learnable Parameter.
        assert isinstance(layer.eps, torch.nn.Parameter)
        assert layer.eps.requires_grad is True


def test_default_enc_eps_matches_pyg_defaults():
    """Defaults preserve the pre-unification pretraining behaviour."""
    model = _make_model()
    assert model.hparams["enc_eps"] == 0.0
    assert model.hparams["enc_train_eps"] is False
    for layer in model.encoder.layers:
        assert layer.eps.requires_grad is False


def test_forward_returns_expected_shapes():
    """End-to-end forward still produces the correct node/graph outputs."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()

    batch = _make_batch(batch_size=2, n_nodes_per_graph=3)
    with torch.no_grad():
        out = model(batch)

    assert set(out.keys()) == {"node", "graph"}
    assert out["node"].shape == (2 * 3, 2)
    assert out["graph"].shape == (2, 1)


def test_per_layer_embeddings_length_matches_num_layers():
    """The base-class hook must return one node-feature tensor per encoder layer."""
    model = _make_model(enc_num_layers=4)
    model.eval()

    batch = _make_batch()
    with torch.no_grad():
        per_layer, _ = model._get_per_layer_embeddings(batch)

    assert isinstance(per_layer, list)
    assert len(per_layer) == 4
