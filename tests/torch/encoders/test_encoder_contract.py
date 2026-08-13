"""Cross-architecture contract for :class:`BaseGraphEncoder` subclasses.

Pins the structural contract introduced by issue #24: every registered
:class:`BaseGraphEncoder` subclass in the migrated set must implement
:meth:`forward_nodes_per_layer` concretely (i.e. not inherit the
``NotImplementedError``-raising default on the base) and must return
``(list[Tensor], Batch)`` with per-layer node features of shape
``[num_nodes, atom_hidden_dim]``.

The purpose of this file is drift prevention. If a future PR adds a new
graph encoder subclass and forgets to implement the pretraining hook, the
enumeration test below will flag it. The 3D variants (``e3gnn``, ``gps3d``,
``gt3d``) share the contract with their 2D counterparts — they read 3D
coordinates from ``graph.pos`` (attached in the fixture below) rather than
from a separate positional argument.
"""

import pytest
import torch


pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402

from matcha.torch.encoders.attentivefp import AttentiveFP  # noqa: E402
from matcha.torch.encoders.base_encoder import EncoderRegistry  # noqa: E402
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder  # noqa: E402
from matcha.torch.encoders.e3gnn import E3GNN  # noqa: E402
from matcha.torch.encoders.gatedgcn import GatedGCN  # noqa: E402
from matcha.torch.encoders.gin import GIN  # noqa: E402
from matcha.torch.encoders.gps import GPS  # noqa: E402
from matcha.torch.encoders.gps3d import GPS3D  # noqa: E402
from matcha.torch.encoders.gt import GT  # noqa: E402
from matcha.torch.encoders.gt3d import GT3D  # noqa: E402


_NUM_LAYERS = 3
_ATOM_INPUT_DIM = 8
_BOND_INPUT_DIM = 4
_ATOM_HIDDEN_DIM = 16
_NUM_HEADS = 4


# Every registered graph encoder is now on the unified contract. Kept as an
# extensibility hook in case a future encoder is landed intentionally
# behind the migration (must document why alongside the alias).
_KNOWN_UNMIGRATED_ALIASES: frozenset[str] = frozenset()


# Common encoder kwargs — every migrated encoder accepts these.
_COMMON_KWARGS = dict(
    num_layers=_NUM_LAYERS,
    atom_input_dim=_ATOM_INPUT_DIM,
    bond_input_dim=_BOND_INPUT_DIM,
    atom_hidden_dim=_ATOM_HIDDEN_DIM,
    dropout=0.0,
    jk="last",
    readout="sum",
    laplacian_k=0,
    rwse_k=0,
    elstatic_k=0,
    distmat_k=0,
    rrwp_k=0,
)


# Per-class extras needed by each concrete encoder's __init__ signature.
_ENCODER_EXTRAS: dict[type, dict] = {
    GIN: dict(
        activation="relu",
        aggregation="sum",
        norm=None,
        eps=0.0,
        train_eps=False,
    ),
    GatedGCN: dict(activation="relu", norm=None),
    GPS: dict(
        activation="relu",
        norm="adarmsn",
        num_heads=_NUM_HEADS,
        expansion_k=1,
        distance_k=None,
    ),
    GT: dict(
        activation="relu",
        num_heads=_NUM_HEADS,
        expansion_k=1,
        distance_k=None,
    ),
    AttentiveFP: dict(),
    E3GNN: dict(
        m_dim=8,
        fourier_features=2,
        soft_edge=False,
        norm_feats=False,
        norm_coors=False,
        update_coors=True,
        activation="relu",
        coor_weights_clamp_value=100.0,
        norm_coors_scale_init=1e-2,
    ),
    GPS3D: dict(
        raw_atom_input_dim=_ATOM_INPUT_DIM,
        num_heads=_NUM_HEADS,
        expansion_k=1,
        num_kernels=4,
        activation="relu",
        norm="adarmsn",
    ),
    GT3D: dict(
        raw_atom_input_dim=_ATOM_INPUT_DIM,
        num_heads=_NUM_HEADS,
        expansion_k=1,
        num_kernels=4,
        activation="relu",
    ),
}


def _build_encoder(encoder_cls: type) -> BaseGraphEncoder:
    """Instantiate ``encoder_cls`` with the shared tiny-encoder settings."""
    return encoder_cls(**_COMMON_KWARGS, **_ENCODER_EXTRAS[encoder_cls])


def _make_batch(batch_size: int = 2) -> Batch:
    """Small PyG batch of chain graphs with random node/edge features.

    Includes 3D coordinates on ``pos`` so 3D encoders (E3GNN, GPS3D, GT3D)
    can consume the same fixture as their 2D counterparts. Attaching
    ``pos`` is harmless for encoders that ignore it.
    """
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
                pos=torch.randn(n_nodes, 3),
            )
        )
    return Batch.from_data_list(graphs)


def _registered_graph_encoders() -> list[tuple[str, type]]:
    """Return ``(alias, cls)`` pairs for every registered BaseGraphEncoder."""
    seen: dict[type, str] = {}
    for alias, cls in EncoderRegistry.items():
        if not isinstance(cls, type):
            continue
        if not issubclass(cls, BaseGraphEncoder):
            continue
        seen.setdefault(cls, alias)
    return sorted(
        ((alias, cls) for cls, alias in seen.items()),
        key=lambda pair: pair[1].__name__,
    )


_MIGRATED_ENCODERS = [
    cls for _, cls in _registered_graph_encoders() if cls in _ENCODER_EXTRAS
]


def test_migrated_set_matches_encoder_extras():
    """Guardrail: the manually maintained ``_ENCODER_EXTRAS`` table must
    line up with the migrated set discovered from the registry.

    If a new graph encoder is registered and this table is not updated, the
    parametrized shape/dtype test below cannot cover it. This test catches
    that oversight explicitly rather than silently under-covering.
    """
    from_registry = {
        cls
        for alias, cls in _registered_graph_encoders()
        if alias not in _KNOWN_UNMIGRATED_ALIASES
    }
    from_table = set(_ENCODER_EXTRAS.keys())
    missing = from_registry - from_table
    extra = from_table - from_registry
    assert not missing, (
        "New BaseGraphEncoder subclass(es) registered without an "
        f"_ENCODER_EXTRAS entry: {sorted(c.__name__ for c in missing)}. "
        "Add them here or list their alias in _KNOWN_UNMIGRATED_ALIASES."
    )
    assert not extra, (
        "_ENCODER_EXTRAS has entries for classes not registered on "
        f"EncoderRegistry: {sorted(c.__name__ for c in extra)}."
    )


def test_all_registered_graph_encoders_override_forward_nodes_per_layer():
    """Every migrated :class:`BaseGraphEncoder` subclass must override the
    per-layer node hook.

    The base class provides a ``NotImplementedError``-raising default so
    the pretraining path fails loudly against un-migrated encoders. This
    test asserts that no encoder in the migrated set falls back to that
    default, so the pretraining hook is guaranteed to work.
    """
    missing: list[str] = []
    for alias, cls in _registered_graph_encoders():
        if alias in _KNOWN_UNMIGRATED_ALIASES:
            continue
        if cls.forward_nodes_per_layer is BaseGraphEncoder.forward_nodes_per_layer:
            missing.append(cls.__name__)

    assert not missing, (
        "BaseGraphEncoder subclasses missing a concrete "
        f"forward_nodes_per_layer override: {missing}. Either implement the "
        "hook or add the alias to _KNOWN_UNMIGRATED_ALIASES with a comment."
    )


@pytest.mark.parametrize("encoder_cls", _MIGRATED_ENCODERS, ids=lambda c: c.__name__)
def test_forward_nodes_per_layer_returns_list_and_batch(
    encoder_cls: type,
):
    """Shape/type contract: ``forward_nodes_per_layer`` returns a list of
    per-layer node tensors and a PyG :class:`Batch`.

    Runs against every registered + migrated :class:`BaseGraphEncoder`
    subclass, so the contract is enforced structurally rather than only
    per-encoder in the individual test files.
    """
    torch.manual_seed(0)
    encoder = _build_encoder(encoder_cls)
    encoder.eval()

    with torch.no_grad():
        result = encoder.forward_nodes_per_layer(_make_batch())

    assert isinstance(result, tuple) and len(result) == 2, (
        f"{encoder_cls.__name__}.forward_nodes_per_layer must return a "
        "2-tuple (list[Tensor], Batch)."
    )
    per_layer, g = result

    assert isinstance(per_layer, list)
    assert len(per_layer) == _NUM_LAYERS
    assert isinstance(g, Batch)

    total_nodes = int(g.batch.numel())
    for feats in per_layer:
        assert isinstance(feats, torch.Tensor)
        assert feats.dtype == torch.float32
        assert feats.shape == (total_nodes, _ATOM_HIDDEN_DIM)
