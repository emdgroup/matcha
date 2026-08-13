"""Tests that pretraining models consume canonical graph encoders.

Focus of these tests is the Stage-3 unification from issue #24: after deleting
``GatedGCNPretrainingEncoder``, ``GPSPretrainingEncoder``, ``GTPretrainingEncoder``,
and ``AttentiveFPPretrainingEncoder``, each pretraining model must:

- wire ``_build_encoder`` to the canonical encoder in ``matcha.torch.encoders``;
- inherit ``_get_per_layer_embeddings`` from :class:`BaseGraphPretrainingModel`
  (delegating to the canonical encoder's ``forward_nodes_per_layer``);
- still produce the expected node/graph prediction shapes end-to-end.
"""

import pytest
import torch


pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402

from matcha.datamodules.classic.graph_datamodule import (  # noqa: E402
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
)
from matcha.torch.encoders.attentivefp import AttentiveFP  # noqa: E402
from matcha.torch.encoders.gatedgcn import GatedGCN  # noqa: E402
from matcha.torch.encoders.gps import GPS  # noqa: E402
from matcha.torch.encoders.gt import GT  # noqa: E402
from matcha.torch.models.pretraining.attentivefp_pretraining import (  # noqa: E402
    AttentiveFPPretraining,
)
from matcha.torch.models.pretraining.gatedgcn_pretraining import (  # noqa: E402
    GatedGCNPretraining,
)
from matcha.torch.models.pretraining.gps_pretraining import GPSPretraining  # noqa: E402
from matcha.torch.models.pretraining.gt_pretraining import GTPretraining  # noqa: E402


ATOM_HIDDEN_DIM = 16
NUM_LAYERS = 2


_COMMON_KWARGS = dict(
    num_node_targets=2,
    num_graph_targets=1,
    enc_num_layers=NUM_LAYERS,
    enc_atom_hidden_dim=ATOM_HIDDEN_DIM,
    enc_readout="sum",
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


# Per-architecture overrides that reflect each model's required parameters
# (e.g. attention heads, activation, norm choice).
_ARCHITECTURES = [
    pytest.param(
        GatedGCNPretraining,
        GatedGCN,
        dict(enc_activation="relu", enc_norm=None, enc_jk="last"),
        id="gatedgcn",
    ),
    pytest.param(
        GPSPretraining,
        GPS,
        dict(
            enc_activation="relu",
            enc_norm="layer",
            enc_jk="last",
            enc_num_heads=4,
            enc_expansion_k=1,
            enc_distance_k=None,
        ),
        id="gps",
    ),
    pytest.param(
        GTPretraining,
        GT,
        dict(
            enc_activation="relu",
            enc_jk="last",
            enc_num_heads=4,
            enc_expansion_k=1,
            enc_distance_k=None,
        ),
        id="gt",
    ),
    pytest.param(
        AttentiveFPPretraining,
        AttentiveFP,
        dict(enc_jk="last"),
        id="attentivefp",
    ),
]


def _build_model(model_cls, extra_kwargs):
    """Build a tiny pretraining model with positional encodings disabled."""
    kwargs = dict(_COMMON_KWARGS)
    kwargs.update(extra_kwargs)
    return model_cls(**kwargs)


def _make_batch(batch_size: int = 2, n_nodes_per_graph: int = 3):
    """Minimal batch dict for the pretraining model forward pass."""
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


@pytest.mark.parametrize("model_cls, encoder_cls, extra_kwargs", _ARCHITECTURES)
def test_encoder_is_canonical(model_cls, encoder_cls, extra_kwargs):
    """The pretraining model instantiates the canonical encoder class."""
    model = _build_model(model_cls, extra_kwargs)
    assert isinstance(model.encoder, encoder_cls)


@pytest.mark.parametrize("model_cls, encoder_cls, extra_kwargs", _ARCHITECTURES)
def test_per_layer_embeddings_length_matches_num_layers(
    model_cls, encoder_cls, extra_kwargs
):
    """Base-class hook returns one node-feature tensor per encoder layer."""
    model = _build_model(model_cls, extra_kwargs)
    model.eval()

    batch = _make_batch()
    with torch.no_grad():
        per_layer, _ = model._get_per_layer_embeddings(batch)

    assert isinstance(per_layer, list)
    assert len(per_layer) == NUM_LAYERS


@pytest.mark.parametrize("model_cls, encoder_cls, extra_kwargs", _ARCHITECTURES)
def test_forward_returns_expected_shapes(model_cls, encoder_cls, extra_kwargs):
    """End-to-end forward returns the correct node and graph prediction shapes."""
    torch.manual_seed(0)
    model = _build_model(model_cls, extra_kwargs)
    model.eval()

    batch = _make_batch(batch_size=2, n_nodes_per_graph=3)
    with torch.no_grad():
        out = model(batch)

    assert set(out.keys()) == {"node", "graph"}
    assert out["node"].shape == (2 * 3, 2)
    assert out["graph"].shape == (2, 1)
