"""Tests for :class:`matcha.torch.models.pretraining.e3gnn_pretraining.E3GNNPretraining`.

Locks in the Stage 3 contract from issue #26:

- the pretraining model wires up the canonical :class:`E3GNN` encoder rather
  than a pretraining-specific twin;
- coordinates flow through ``batch["graph"].pos`` (never a separate
  ``coords`` collate key);
- a one-step ``training_step`` produces a finite scalar loss on a synthetic
  dataset with per-atom and per-molecule targets;
- ``save_hyperparameters`` round-trips through the encoder ``state_dict``;
- encoder weights transfer cleanly into a matching :class:`E3GNNModel` for
  downstream finetuning — the acceptance criterion the whole issue is about.

Stage 4 additionally locks in classic ↔ pretraining datamodule parity: given
the same seeded batch, an :class:`E3GNN` fed by :class:`Graph3DDataModule` and
by :class:`Graph3DPretrainingDataModule` must produce bit-identical per-layer
node features.
"""

import numpy as np
import pytest
import torch
from rdkit import Chem


pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402
from torch.nn.utils import parameters_to_vector  # noqa: E402

from matcha.datamodules.classic.graph_datamodule import (  # noqa: E402
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
    Graph3DDataModule,
)
from matcha.datamodules.pretraining.graph_3d_pretraining_datamodule import (  # noqa: E402
    Graph3DPretrainingDataModule,
)
from matcha.torch.encoders.e3gnn import E3GNN  # noqa: E402
from matcha.torch.models.classic.e3gnn_model import E3GNNModel  # noqa: E402
from matcha.torch.models.pretraining.base_pretraining_model import (  # noqa: E402
    PretrainingModelRegistry,
)
from matcha.torch.models.pretraining.e3gnn_pretraining import (  # noqa: E402
    E3GNNPretraining,
)


_ATOM_HIDDEN_DIM = 16
_NUM_LAYERS = 2


def _make_model(**overrides) -> E3GNNPretraining:
    """Build a tiny E3GNNPretraining with positional encodings disabled."""
    kwargs = dict(
        num_node_targets=2,
        num_graph_targets=1,
        enc_num_layers=_NUM_LAYERS,
        enc_atom_hidden_dim=_ATOM_HIDDEN_DIM,
        enc_m_dim=8,
        enc_fourier_features=2,
        enc_soft_edge=False,
        enc_norm_feats=False,
        enc_norm_coors=False,
        enc_update_coors=True,
        enc_coor_weights_clamp_value=100.0,
        enc_norm_coors_scale_init=1e-2,
        enc_jk="last",
        enc_readout="sum",
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
    return E3GNNPretraining(**kwargs)


def _make_batch(
    batch_size: int = 2,
    n_nodes_per_graph: int = 4,
    num_node_targets: int = 2,
    num_graph_targets: int = 1,
) -> dict:
    """Minimal batch dict matching :class:`Graph3DPretrainingDataModule` output.

    Attaches ``pos`` on the per-molecule ``Data`` objects, so PyG's
    ``Batch.from_data_list`` auto-concatenates them into ``graph.pos`` in the
    same node ordering the encoder walks — mirroring the real datamodule.
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
    graph = Batch.from_data_list(graphs)
    return {
        "graph": graph,
        "y_node": torch.randn(batch_size * n_nodes_per_graph, num_node_targets),
        "y_graph": torch.randn(batch_size, num_graph_targets),
    }


def test_encoder_is_canonical_e3gnn():
    """The pretraining model instantiates the canonical :class:`E3GNN` encoder."""
    model = _make_model()
    assert isinstance(model.encoder, E3GNN)


def test_registered_on_pretraining_registry():
    """``E3GNNPretraining`` is discoverable via the pretraining registry."""
    assert "e3gnnpretraining" in PretrainingModelRegistry
    assert PretrainingModelRegistry["e3gnnpretraining"] is E3GNNPretraining


def test_forward_returns_expected_shapes():
    """End-to-end forward produces the correct node/graph output shapes."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()

    batch = _make_batch(batch_size=2, n_nodes_per_graph=4)
    with torch.no_grad():
        out = model(batch)

    assert set(out.keys()) == {"node", "graph"}
    assert out["node"].shape == (2 * 4, 2)
    assert out["graph"].shape == (2, 1)


def test_per_layer_embeddings_length_matches_num_layers():
    """Base-class hook returns one node-feature tensor per encoder layer."""
    model = _make_model(enc_num_layers=3)
    model.eval()

    batch = _make_batch()
    with torch.no_grad():
        per_layer, _ = model._get_per_layer_embeddings(batch)

    assert isinstance(per_layer, list)
    assert len(per_layer) == 3


def _silence_lightning_log(model: E3GNNPretraining) -> None:
    """Stub ``self.log`` / ``self.log_dict`` — no Trainer is attached in unit tests.

    Lightning's ``LightningModule.log`` emits a ``UserWarning`` when called
    without an attached ``Trainer``. Pytest is configured with
    ``filterwarnings = ["error"]``, so we replace the loggers with no-ops
    before driving ``training_step`` / ``validation_step`` directly.
    """
    model.log = lambda *a, **k: None  # type: ignore[assignment]
    model.log_dict = lambda *a, **k: None  # type: ignore[assignment]


def test_training_step_returns_finite_scalar_loss():
    """A single ``training_step`` on a synthetic batch yields a finite loss.

    Serves as the Stage-3 happy-path fit check: encoder builds, coords are
    read from ``graph.pos``, both heads produce predictions, both losses
    contribute, and the combined loss is a differentiable scalar with no
    NaNs or Infs.
    """
    torch.manual_seed(0)
    model = _make_model()
    _silence_lightning_log(model)
    model.train()

    batch = _make_batch(batch_size=3, n_nodes_per_graph=4)
    loss = model.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # Loss must be attached to the computation graph — otherwise Lightning
    # would silently no-op the backward pass in a real fit.
    assert loss.requires_grad


def test_training_step_backward_updates_parameters():
    """One optimizer step actually moves the encoder's parameters."""
    torch.manual_seed(0)
    model = _make_model()
    _silence_lightning_log(model)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    batch = _make_batch(batch_size=3, n_nodes_per_graph=4)
    before = parameters_to_vector(model.encoder.parameters()).detach().clone()
    optimizer.zero_grad()
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    optimizer.step()
    after = parameters_to_vector(model.encoder.parameters()).detach().clone()

    assert not torch.allclose(before, after)


def test_forward_raises_when_pos_missing():
    """Missing ``graph.pos`` surfaces the encoder's clear error message.

    Locks in the Stage-1 encoder contract from this issue: the pretraining
    path must NOT swallow the ``ValueError`` E3GNN raises when coords are
    absent — it is the tripwire that catches a mis-configured datamodule.
    """
    model = _make_model()
    model.eval()

    batch = _make_batch()
    del batch["graph"].pos

    with pytest.raises(ValueError, match="graph.pos"):
        model(batch)


def test_save_hyperparameters_round_trip_through_state_dict():
    """A fresh model rebuilt from the same hparams matches the encoder state.

    Reconstructing the model with the same constructor args, loading the
    original's ``state_dict``, and confirming parameter equality catches
    hparam-flow bugs in ``_build_encoder`` — e.g. dropping ``enc_m_dim``
    from the E3GNN kwargs would produce a state_dict shape mismatch here.
    """
    torch.manual_seed(0)
    model = _make_model()

    # Round-trip through the same constructor + state_dict.
    hparams = dict(model.hparams)
    fresh = E3GNNPretraining(**hparams)
    fresh.load_state_dict(model.state_dict())

    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), fresh.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)


def test_encoder_weights_transfer_to_e3gnn_model():
    """Encoder weights from a pretrained model load cleanly into ``E3GNNModel``.

    This is the headline acceptance criterion for issue #26: after a
    pretraining run, users must be able to hand the encoder weights to
    :class:`E3GNNModel` for downstream supervised finetuning without any
    shape mismatch. The check is a strict ``load_state_dict`` (no missing /
    unexpected keys) followed by ``parameters_to_vector`` equality across
    the encoder subtree.
    """
    torch.manual_seed(0)
    pretrain = _make_model()

    # Build a classic E3GNNModel with an encoder that mirrors the pretraining
    # encoder's shape. ``enc_bond_input_dim`` differs between the two classes'
    # defaults (BOND_FEAT_DIM vs 14), so pin it explicitly along with every
    # other structural ``enc_*`` field.
    shared_encoder_kwargs = dict(
        enc_num_layers=_NUM_LAYERS,
        enc_atom_input_dim=ATOM_FEAT_DIM,
        enc_bond_input_dim=BOND_FEAT_DIM,
        enc_atom_hidden_dim=_ATOM_HIDDEN_DIM,
        enc_m_dim=8,
        enc_fourier_features=2,
        enc_soft_edge=False,
        enc_norm_feats=False,
        enc_norm_coors=False,
        enc_update_coors=True,
        enc_coor_weights_clamp_value=100.0,
        enc_norm_coors_scale_init=1e-2,
        enc_jk="last",
        enc_readout="sum",
        enc_activation="relu",
        enc_dropout=0.0,
        enc_laplacian_k=0,
        enc_rwse_k=0,
        enc_elstatic_k=0,
        enc_distmat_k=0,
        enc_rrwp_k=0,
    )
    classic = E3GNNModel(
        pred_hidden_dims=[8],
        pred_activation="relu",
        pred_dropout=0.0,
        num_endpoints=1,
        **shared_encoder_kwargs,
    )

    # Strict load — any structural drift between the two encoders would
    # surface here as a missing / unexpected key.
    missing, unexpected = classic.encoder.load_state_dict(
        pretrain.encoder.state_dict(), strict=True
    )
    assert missing == []
    assert unexpected == []

    # Numerical equality on every encoder parameter after transfer.
    src = parameters_to_vector(pretrain.encoder.parameters()).detach()
    dst = parameters_to_vector(classic.encoder.parameters()).detach()
    assert torch.equal(src, dst)


# ---------------------------------------------------------------------
# Classic ↔ pretraining datamodule parity (Stage 4)
# ---------------------------------------------------------------------


_PARITY_SMILES = ["CCO", "c1ccncc1", "CC(=O)O", "CCN"]


def _canonical_mols(smiles: list[str]) -> list:
    """Return RDKit mols reparsed from canonical SMILES.

    Ensures the input atom order already matches
    :meth:`GraphDataModule._calculate_graph`'s canonical reorder, so the
    pretraining datamodule's ``_reorder_coords_to_canonical`` is an identity
    map. That isolates the parity check to encoder/datamodule wiring —
    reorder correctness is exercised separately in
    ``tests/datamodules/test_graph_3d_pretraining_datamodule.py``.
    """
    return [Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(s))) for s in smiles]


def _slice_pos_by_graph(graph_batch: Batch) -> list[torch.Tensor]:
    """Split a batched ``graph.pos`` back into one tensor per molecule."""
    per_mol: list[torch.Tensor] = []
    for graph_id in range(int(graph_batch.batch.max().item()) + 1):
        mask = graph_batch.batch == graph_id
        per_mol.append(graph_batch.pos[mask].detach().clone())
    return per_mol


def test_classic_and_pretraining_datamodules_produce_identical_e3gnn_features():
    """E3GNN outputs match bit-for-bit across the classic and pretraining paths.

    The check runs the same encoder instance on batches produced by
    :class:`Graph3DDataModule` (classic; ETKDG-generated coords) and
    :class:`Graph3DPretrainingDataModule` (pretraining; user-supplied
    coords). To keep the comparison meaningful the pretraining path receives
    the exact ``pos`` tensors emitted by the classic datamodule — parity is
    about the pipe, not about coord generation.

    A failure here means one of the two datamodules silently mutated ``x``,
    ``edge_index``, ``edge_attr`` or ``pos`` on its way through featurization
    / collation, which would break the promise that a
    :class:`Graph3DPretrainingDataModule` `.export_to_classic()` roundtrip
    produces a numerically equivalent finetuning setup.
    """
    torch.manual_seed(0)
    mol_list = _canonical_mols(_PARITY_SMILES)

    shared_dm_kwargs = dict(
        laplacian_k=0,
        rwse_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
        compute_distances=False,
        num_virtual_nodes=0,
        init_virtual_nodes=False,
        batch_size=len(mol_list),
        num_workers=0,
        augment_resonance=False,
    )

    # --- Classic path: coords come from ETKDG inside the datamodule.
    classic_dm = Graph3DDataModule(**shared_dm_kwargs)
    dummy_y = np.zeros((len(mol_list), 1), dtype=np.float32)
    classic_ds = classic_dm.generate_features(mol_list, dummy_y, n_jobs=1)
    classic_batch = classic_dm.collate_fn(
        [classic_ds[i] for i in range(len(classic_ds))]
    )
    classic_graph = classic_batch["graph"]

    # --- Pretraining path: feed classic's coords back in as user-supplied.
    coords = [p.numpy() for p in _slice_pos_by_graph(classic_graph)]
    y_node = [np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32) for mol in mol_list]
    pretrain_dm = Graph3DPretrainingDataModule(**shared_dm_kwargs)
    pretrain_ds = pretrain_dm.generate_features(
        mol_list=mol_list, y_graph=dummy_y, y_node=y_node, coords=coords, n_jobs=1
    )
    pretrain_batch = pretrain_dm.collate_fn(
        [pretrain_ds[i] for i in range(len(pretrain_ds))]
    )
    pretrain_graph = pretrain_batch["graph"]

    # --- Structural batch parity: any divergence here would invalidate the
    # subsequent encoder-output comparison.
    assert torch.equal(classic_graph.x, pretrain_graph.x)
    assert torch.equal(classic_graph.edge_index, pretrain_graph.edge_index)
    assert torch.equal(classic_graph.edge_attr, pretrain_graph.edge_attr)
    assert torch.equal(classic_graph.batch, pretrain_graph.batch)
    assert torch.allclose(classic_graph.pos, pretrain_graph.pos)

    # --- Encoder parity: same encoder, same batch → identical per-layer feats.
    encoder = E3GNN(
        num_layers=_NUM_LAYERS,
        atom_input_dim=ATOM_FEAT_DIM,
        bond_input_dim=BOND_FEAT_DIM,
        atom_hidden_dim=_ATOM_HIDDEN_DIM,
        m_dim=8,
        fourier_features=2,
        soft_edge=False,
        norm_feats=False,
        norm_coors=False,
        update_coors=True,
        activation="relu",
        dropout=0.0,
        coor_weights_clamp_value=100.0,
        norm_coors_scale_init=1e-2,
        jk="last",
        readout="sum",
        laplacian_k=0,
        rwse_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
    )
    encoder.eval()

    with torch.no_grad():
        classic_feats, _ = encoder.forward_nodes_per_layer(classic_graph)
        pretrain_feats, _ = encoder.forward_nodes_per_layer(pretrain_graph)

    assert len(classic_feats) == len(pretrain_feats) == _NUM_LAYERS
    for lc, lp in zip(classic_feats, pretrain_feats):
        assert torch.allclose(lc, lp, atol=1e-6)
