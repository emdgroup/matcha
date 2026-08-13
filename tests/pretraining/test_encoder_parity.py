"""Parity between classic and pretraining encoders (issue #24).

For every graph architecture with a ``*Model`` (classic) and ``*Pretraining``
pair, both paths must build the same canonical encoder from
``matcha.torch.encoders`` when given matching ``enc_*`` hyperparameters.
These tests pin that structural equivalence so future PRs that reintroduce
a pretraining-specific encoder class fail loudly.

RoFormer is intentionally out of scope here — its classic/pretraining parity
is verified separately by :mod:`tests.pretraining.test_roformer_mlm` and
:mod:`tests.torch.encoders.test_roformer`, and this file is scoped to the
five graph architectures per the issue #24 plan.
"""

import pytest
import torch


pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402

from matcha.datamodules.classic.graph_datamodule import (  # noqa: E402
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
)
from matcha.torch.models.classic.attentivefp_model import (  # noqa: E402
    AttentiveFPModel,
)
from matcha.torch.models.classic.e3gnn_model import E3GNNModel  # noqa: E402
from matcha.torch.models.classic.gatedgcn_model import GatedGCNModel  # noqa: E402
from matcha.torch.models.classic.gin_model import GINModel  # noqa: E402
from matcha.torch.models.classic.gps_model import GPSModel  # noqa: E402
from matcha.torch.models.classic.gt_model import GTModel  # noqa: E402
from matcha.torch.models.pretraining.attentivefp_pretraining import (  # noqa: E402
    AttentiveFPPretraining,
)
from matcha.torch.models.pretraining.e3gnn_pretraining import (  # noqa: E402
    E3GNNPretraining,
)
from matcha.torch.models.pretraining.gatedgcn_pretraining import (  # noqa: E402
    GatedGCNPretraining,
)
from matcha.torch.models.pretraining.gin_pretraining import (  # noqa: E402
    GINPretraining,
)
from matcha.torch.models.pretraining.gps_pretraining import (  # noqa: E402
    GPSPretraining,
)
from matcha.torch.models.pretraining.gt_pretraining import GTPretraining  # noqa: E402


_ATOM_HIDDEN_DIM = 16
_NUM_LAYERS = 2


# Encoder hyperparameters shared by every architecture in this file.
# ``enc_atom_input_dim`` and ``enc_bond_input_dim`` are pinned explicitly
# because the classic-model defaults (``44`` / ``14``) do not match the
# pretraining-model defaults (``ATOM_FEAT_DIM=72`` / ``BOND_FEAT_DIM=14``),
# and this file only cares about parity for identical hyperparameters.
_SHARED_ENC_KWARGS = dict(
    enc_num_layers=_NUM_LAYERS,
    enc_atom_input_dim=ATOM_FEAT_DIM,
    enc_bond_input_dim=BOND_FEAT_DIM,
    enc_atom_hidden_dim=_ATOM_HIDDEN_DIM,
    enc_readout="sum",
    enc_dropout=0.0,
    enc_jk="last",
    enc_laplacian_k=0,
    enc_rwse_k=0,
    enc_elstatic_k=0,
    enc_distmat_k=0,
    enc_rrwp_k=0,
)


# Per-architecture ``enc_*`` extras — mirror the concrete encoder's __init__.
_ARCHITECTURES = [
    pytest.param(
        GINModel,
        GINPretraining,
        dict(
            enc_activation="relu",
            enc_aggregation="sum",
            enc_norm=None,
            enc_eps=0.0,
            enc_train_eps=False,
        ),
        id="gin",
    ),
    pytest.param(
        GatedGCNModel,
        GatedGCNPretraining,
        dict(enc_activation="relu", enc_norm=None),
        id="gatedgcn",
    ),
    pytest.param(
        GPSModel,
        GPSPretraining,
        dict(
            enc_activation="relu",
            enc_norm="layer",
            enc_num_heads=4,
            enc_expansion_k=1,
            enc_distance_k=None,
        ),
        id="gps",
    ),
    pytest.param(
        GTModel,
        GTPretraining,
        dict(
            enc_activation="relu",
            enc_num_heads=4,
            enc_expansion_k=1,
            enc_distance_k=None,
        ),
        id="gt",
    ),
    pytest.param(
        AttentiveFPModel,
        AttentiveFPPretraining,
        dict(),
        id="attentivefp",
    ),
    pytest.param(
        E3GNNModel,
        E3GNNPretraining,
        dict(
            enc_activation="relu",
            enc_m_dim=8,
            enc_fourier_features=2,
            enc_soft_edge=False,
            enc_norm_feats=False,
            enc_norm_coors=False,
            enc_update_coors=True,
            enc_coor_weights_clamp_value=100.0,
            enc_norm_coors_scale_init=1e-2,
        ),
        id="e3gnn",
    ),
]


def _make_batch(batch_size: int = 2, n_nodes_per_graph: int = 3) -> Batch:
    """Small PyG batch with realistic atom/bond feature dimensions.

    Every graph carries ``pos`` (3D coords) so the batch is valid input for
    :class:`E3GNN` alongside the 2D encoders — the latter ignore ``pos``, so
    attaching it unconditionally keeps the parametrization one-shape-fits-all.
    """
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
                pos=torch.randn(n_nodes_per_graph, 3),
            )
        )
    return Batch.from_data_list(graphs)


def _build_pair(classic_cls, pretrain_cls, extra_kwargs):
    """Build a classic + pretraining model with identical ``enc_*`` kwargs.

    The classic model uses tiny predictor dims and a single endpoint; the
    pretraining model uses tiny node/graph heads with a single target each.
    Only the encoder is exercised by the parity assertions below, so head /
    predictor sizes are irrelevant.
    """
    common = dict(_SHARED_ENC_KWARGS)
    common.update(extra_kwargs)
    classic = classic_cls(
        pred_hidden_dims=[8],
        pred_dropout=0.0,
        pred_activation="relu",
        num_endpoints=1,
        **common,
    )
    pretrain = pretrain_cls(
        num_node_targets=1,
        num_graph_targets=1,
        node_head_dims=[8],
        graph_head_dims=[8],
        pred_activation="relu",
        pred_dropout=0.0,
        **common,
    )
    return classic, pretrain


@pytest.mark.parametrize("classic_cls, pretrain_cls, extra_kwargs", _ARCHITECTURES)
def test_named_parameter_keys_match(classic_cls, pretrain_cls, extra_kwargs):
    """The classic and pretraining encoders share exactly the same parameter tree.

    Different keys would mean one path built a differently-shaped module
    graph (e.g. an extra norm, a missing residual, a swapped layer), which
    is the class of drift issue #24 is designed to eliminate.
    """
    classic, pretrain = _build_pair(classic_cls, pretrain_cls, extra_kwargs)
    classic_keys = dict(classic.encoder.named_parameters()).keys()
    pretrain_keys = dict(pretrain.encoder.named_parameters()).keys()
    assert classic_keys == pretrain_keys


@pytest.mark.parametrize("classic_cls, pretrain_cls, extra_kwargs", _ARCHITECTURES)
def test_module_tree_string_matches(classic_cls, pretrain_cls, extra_kwargs):
    """``repr`` of the two encoders is identical.

    Complements the parameter-key check: two modules can have the same
    parameter names but different intermediate module wiring (e.g. an
    ``nn.Identity`` swapped for an ``nn.LayerNorm`` in the same position);
    ``str()`` on the module tree surfaces that.
    """
    classic, pretrain = _build_pair(classic_cls, pretrain_cls, extra_kwargs)
    assert str(classic.encoder) == str(pretrain.encoder)


@pytest.mark.parametrize("classic_cls, pretrain_cls, extra_kwargs", _ARCHITECTURES)
def test_encoder_forward_output_allclose(classic_cls, pretrain_cls, extra_kwargs):
    """After copying the classic encoder's weights into the pretraining
    encoder, both must produce numerically identical embeddings for the
    same input batch.

    A ``load_state_dict`` round-trip only succeeds if the two encoders
    share the same parameter names and shapes; the subsequent
    ``allclose`` then verifies the forward wiring (layer order, residuals,
    normalization placement) matches — the exact failure mode that
    produced the original GIN drift between canonical and pretraining
    encoders.
    """
    classic, pretrain = _build_pair(classic_cls, pretrain_cls, extra_kwargs)

    # Sync weights: any key mismatch would have failed the earlier test.
    pretrain.encoder.load_state_dict(classic.encoder.state_dict())

    classic.encoder.eval()
    pretrain.encoder.eval()

    torch.manual_seed(0)
    batch = _make_batch()

    with torch.no_grad():
        classic_out = classic.encoder(batch)
        pretrain_out = pretrain.encoder(batch)

    assert classic_out.shape == pretrain_out.shape
    assert torch.allclose(classic_out, pretrain_out, atol=1e-6)
