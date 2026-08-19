"""Regression tests for nested-Finetuner encoder resolution.

Covers issue #67 (LoRA crash on phase-2 artifacts) and the parallel silent
degradation of full-strategy layer-wise LR decay when ``self.pretrain`` is a
nested :class:`Finetuner`. Uses the ``_SELF_CONTAINED_SENTINEL`` skeleton path
so all fixtures live in memory.

Also covers issue #81: the ``keep_existing_predictor`` flag on
:class:`Finetuner` which, when ``False``, discards the pretrained predictor
end-to-end (including across nested :class:`Finetuner` wrappers) and feeds
the new head from the leaf encoder directly.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from matcha.torch.models.finetuning.finetuner import (
    Finetuner,
    _SELF_CONTAINED_SENTINEL,
    _resolve_leaf_encoder,
)
from matcha.torch.models.finetuning.passthrough_predictor import PassthroughPredictor
from matcha.torch.models.finetuning.pretrained_encoder_wrapper import (
    PretrainedEncoderWrapper,
)


GIN_PRETRAIN_PARAMS = {
    "enc_num_layers": 1,
    "enc_atom_input_dim": 44,
    "enc_bond_input_dim": 14,
    "enc_atom_hidden_dim": 32,
    "enc_aggregation": "sum",
    "enc_jk": "last",
    "enc_norm": None,
    "enc_readout": "sum",
    "enc_activation": "relu",
    "enc_dropout": 0.0,
    "enc_laplacian_k": 0,
    "enc_rwse_k": 0,
    "enc_elstatic_k": 0,
    "enc_distmat_k": 0,
    "enc_rrwp_k": 0,
    "pred_hidden_dims": [32],
    "pred_task_head_dims": None,
    "pred_activation": "relu",
    "pred_dropout": 0.0,
    "num_endpoints": 1,
    "loss_fn": "mse",
    "loss_args": {},
    "optimizer": "adam",
    "optimizer_args": {"lr": 1e-3},
    "scheduler": "cosine_annealing",
    "scheduler_args": {"min_lr": 1e-6, "total_steps": 50},
}

GIN_PRETRAINING_PARAMS = {
    "num_node_targets": 1,
    "num_graph_targets": 1,
    "enc_num_layers": 1,
    "enc_atom_input_dim": 44,
    "enc_bond_input_dim": 14,
    "enc_atom_hidden_dim": 32,
    "enc_aggregation": "sum",
    "enc_jk": "last",
    "enc_norm": None,
    "enc_readout": "sum",
    "enc_activation": "relu",
    "enc_dropout": 0.0,
    "enc_laplacian_k": 0,
    "enc_rwse_k": 0,
    "enc_elstatic_k": 0,
    "enc_distmat_k": 0,
    "enc_rrwp_k": 0,
    "node_head_dims": None,
    "graph_head_dims": None,
    "pred_activation": "relu",
    "pred_dropout": 0.0,
    "loss_fn": "mse",
    "loss_args": {},
    "optimizer": "adam",
    "optimizer_args": {"lr": 1e-4},
    "scheduler": "cosine_annealing",
    "scheduler_args": {"min_lr": 1e-6, "total_steps": 50},
}


NESTED_HPARAMS = {
    "architecture": "ginmodel",
    "pred_hidden_dims": [32],
    "task_head_dims": None,
    "activation": "relu",
    "dropout": 0.0,
    "num_endpoints": 1,
    "loss_fn": "mse",
    "loss_args": {},
    "optimizer": "adam",
    "optimizer_args": {"lr": 1e-4},
    "pretrain_lr": 1e-6,
    "pretrain_decay": 0.5,
    "scheduler": "cosine_annealing",
    "scheduler_args": {"min_lr": 1e-6, "total_steps": 50},
    "finetuning_strategy": "full",
    "lora_rank": 4,
    "lora_alpha": 8.0,
    "lora_min_dim": 32,
}


def _make_nested_finetuner(
    leaf_origin: str,
    finetuning_strategy: str,
    keep_existing_predictor: bool = True,
    nested_keep_existing_predictor: bool = True,
):
    """Build a Finetuner whose ``pretrain`` is itself a Finetuner over ``leaf_origin``.

    ``nested_keep_existing_predictor`` controls the flag baked into the inner
    (nested) Finetuner's hparams — this only matters when the outer flag is
    ``True`` (otherwise the outer strip clobbers the entire chain).
    """
    if leaf_origin == "classic":
        inner_config = {
            "origin_type": "classic",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
        }
        outer_pretrain_params = GIN_PRETRAIN_PARAMS
    elif leaf_origin == "pretraining":
        inner_config = {
            "origin_type": "pretraining",
            "pretrain_params": GIN_PRETRAINING_PARAMS,
            "source_class": "GINPretraining",
        }
        outer_pretrain_params = GIN_PRETRAINING_PARAMS
    else:
        raise ValueError(f"Unknown leaf_origin: {leaf_origin}")

    nested_hparams = dict(NESTED_HPARAMS)
    nested_hparams["keep_existing_predictor"] = nested_keep_existing_predictor

    outer_config = {
        "origin_type": "finetuner",
        "pretrain_params": outer_pretrain_params,
        "source_class": None,
        "nested_pretrain_config": inner_config,
        "nested_hparams": nested_hparams,
    }
    return Finetuner(
        architecture="finetunermodel",
        path_to_pretrained=_SELF_CONTAINED_SENTINEL,
        pred_hidden_dims=[32],
        num_endpoints=1,
        dropout=0.0,
        activation="relu",
        optimizer_args={"lr": 1e-4},
        scheduler_args={"min_lr": 1e-6, "total_steps": 50},
        finetuning_strategy=finetuning_strategy,
        lora_rank=4,
        lora_alpha=8.0,
        lora_min_dim=32,
        keep_existing_predictor=keep_existing_predictor,
        _pretrain_config=outer_config,
    )


def _make_flat_finetuner(
    origin: str,
    finetuning_strategy: str,
    keep_existing_predictor: bool = True,
):
    """Build a non-nested Finetuner (single-hop) over a classic or pretraining origin."""
    if origin == "classic":
        config = {
            "origin_type": "classic",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
        }
        architecture = "ginmodel"
    elif origin == "pretraining":
        config = {
            "origin_type": "pretraining",
            "pretrain_params": GIN_PRETRAINING_PARAMS,
            "source_class": "GINPretraining",
        }
        architecture = "ginmodel"
    else:
        raise ValueError(f"Unknown origin: {origin}")

    return Finetuner(
        architecture=architecture,
        path_to_pretrained=_SELF_CONTAINED_SENTINEL,
        pred_hidden_dims=[32],
        num_endpoints=1,
        dropout=0.0,
        activation="relu",
        optimizer_args={"lr": 1e-4},
        scheduler_args={"min_lr": 1e-6, "total_steps": 50},
        finetuning_strategy=finetuning_strategy,
        lora_rank=4,
        lora_alpha=8.0,
        lora_min_dim=32,
        keep_existing_predictor=keep_existing_predictor,
        _pretrain_config=config,
    )


def _make_graph_batch(batch_size: int = 3) -> Batch:
    """Small PyG batch matching the GIN input contract (44-dim atoms, 14-dim bonds)."""
    graphs = []
    torch.manual_seed(0)
    for _ in range(batch_size):
        n_nodes = 4
        src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
        dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        graphs.append(
            Data(
                x=torch.randn(n_nodes, 44),
                edge_index=edge_index,
                edge_attr=torch.randn(edge_index.size(1), 14),
            )
        )
    return Batch.from_data_list(graphs)


class TestResolveLeafEncoder:
    """Direct coverage of the helper contract."""

    def test_returns_encoder_for_non_nested_pretrain(self):
        classic_config = {
            "origin_type": "classic",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
        }
        model = Finetuner(
            architecture="ginmodel",
            path_to_pretrained=_SELF_CONTAINED_SENTINEL,
            pred_hidden_dims=[32],
            num_endpoints=1,
            dropout=0.0,
            activation="relu",
            optimizer_args={"lr": 1e-4},
            scheduler_args={"min_lr": 1e-6, "total_steps": 50},
            _pretrain_config=classic_config,
        )
        assert _resolve_leaf_encoder(model.pretrain) is model.pretrain.encoder

    def test_returns_encoder_for_classic_pretrain(self):
        nested = _make_nested_finetuner("classic", "full")
        leaf = _resolve_leaf_encoder(nested.pretrain)
        # nested.pretrain is a Finetuner; walking one .pretrain hop reaches the
        # classic GIN model whose .encoder is the leaf.
        assert leaf is nested.pretrain.pretrain.encoder


class TestNestedLoraSetup:
    """Regression: LoRA on a phase-2 artifact must not raise AttributeError."""

    def test_nested_lora_classic_leaf_does_not_raise(self):
        model = _make_nested_finetuner("classic", "lora")
        assert model.pretrain_optimizer is None
        assert model.predictor_optimizer is not None

    def test_nested_lora_pretraining_leaf_does_not_raise(self):
        model = _make_nested_finetuner("pretraining", "lora")
        assert model.pretrain_optimizer is None
        assert model.predictor_optimizer is not None


class TestNestedFullSetupLrDecay:
    """Regression: layer-wise LR decay must reach the leaf encoder."""

    def test_layer_wise_decay_groups_are_built(self):
        model = _make_nested_finetuner("classic", "full")
        pretrain_lr = model.hparams["pretrain_lr"]
        decayed_groups = [
            g for g in model.pretrain_optimizer.param_groups if g["lr"] < pretrain_lr
        ]
        # If the encoder were resolved to the missing `self.pretrain.encoder`
        # (nested case) the entire encoder would land in the catch-all bucket
        # at flat `pretrain_lr` — no group would be strictly below it.
        assert decayed_groups, (
            "No param group has lr < pretrain_lr — layer-wise decay did not "
            "reach the leaf encoder."
        )


# Sanity: confirm module-level helper is idempotent for nn.Module leaves.
def test_resolve_leaf_encoder_no_op_for_bare_module():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)

    wrapper = Wrapper()
    assert _resolve_leaf_encoder(wrapper) is wrapper.encoder


# ---------------------------------------------------------------------------
# Issue #81: keep_existing_predictor flag
# ---------------------------------------------------------------------------


def _collect_predictors(model: Finetuner) -> list[nn.Module]:
    """Walk the pretrain chain and return each level's ``.predictor`` module."""
    predictors: list[nn.Module] = []
    current = model.pretrain
    while isinstance(current, Finetuner):
        predictors.append(current.predictor)
        current = current.pretrain
    predictors.append(current.predictor)
    return predictors


class TestKeepExistingPredictorFlatClassic:
    """Flat (single-hop) classic origin — the primary case in the issue."""

    @pytest.mark.parametrize("strategy", ["full", "lora"])
    def test_false_installs_passthrough_and_uses_encoder_fp_dim(self, strategy):
        model = _make_flat_finetuner("classic", strategy, keep_existing_predictor=False)

        # Leaf's predictor was stripped end-to-end.
        assert isinstance(model.pretrain.predictor, PassthroughPredictor)
        # pretrain_output_dim now matches the leaf encoder's fp_dim (raw output),
        # not the pretrained predictor's last hidden dim.
        leaf = _resolve_leaf_encoder(model.pretrain)
        assert model.pretrain_output_dim == leaf.fp_dim
        # No pretrained-predictor parameters remain in the module tree.
        assert sum(p.numel() for p in model.pretrain.predictor.parameters()) == 0
        # The new head still consumes the correct input width.
        assert model.predictor.layers[0].in_features == leaf.fp_dim

    def test_true_preserves_existing_behavior(self):
        model = _make_flat_finetuner("classic", "full", keep_existing_predictor=True)
        # Predictor object is intact (still an MLP, not a passthrough) with its
        # hidden layers wired up; only the prediction_head has been dropped.
        assert not isinstance(model.pretrain.predictor, PassthroughPredictor)
        assert model.pretrain.predictor.prediction_head is None
        # pretrain_output_dim equals the pretrained predictor's last hidden dim
        # (32 from GIN_PRETRAIN_PARAMS["pred_hidden_dims"] = [32]), not fp_dim.
        assert model.pretrain_output_dim == model.pretrain.latent_dim

    def test_false_full_setup_has_no_pretrained_predictor_params(self):
        model = _make_flat_finetuner("classic", "full", keep_existing_predictor=False)
        # PassthroughPredictor has zero parameters — assert defensively that no
        # id() from the stripped predictor slipped into any optimizer group.
        stripped_ids = {id(p) for p in model.pretrain.predictor.parameters()}
        for group in model.pretrain_optimizer.param_groups:
            group_ids = {id(p) for p in group["params"]}
            assert stripped_ids.isdisjoint(group_ids)

    def test_forward_pass_shape_under_false(self):
        model = _make_flat_finetuner("classic", "full", keep_existing_predictor=False)
        model.eval()
        batch = {"graph": _make_graph_batch(batch_size=3)}
        with torch.no_grad():
            out = model.forward(batch)
        assert out.shape == (3, 1)


class TestKeepExistingPredictorPretrainingOrigin:
    """Pure pretraining origin — the flag is a no-op (identical predictions)."""

    @pytest.mark.parametrize("strategy", ["full", "lora"])
    def test_flag_is_noop_for_predictions(self, strategy):
        model_keep = _make_flat_finetuner(
            "pretraining", strategy, keep_existing_predictor=True
        )
        model_strip = _make_flat_finetuner(
            "pretraining", strategy, keep_existing_predictor=False
        )

        # Both wrap a PretrainedEncoderWrapper — no hidden predictor to strip.
        assert isinstance(model_keep.pretrain, PretrainedEncoderWrapper)
        assert isinstance(model_strip.pretrain, PretrainedEncoderWrapper)
        # Both surface the same output width.
        assert model_keep.pretrain_output_dim == model_strip.pretrain_output_dim
        # Neither the leaf's predictor nor the module tree carry any old
        # predictor parameters — both models have identical parameter shapes.
        assert set(model_keep.state_dict().keys()) == set(
            model_strip.state_dict().keys()
        )

        # Copy weights (encoder + new head) across so the two models compute
        # the same downstream projection.
        model_strip.load_state_dict(model_keep.state_dict())
        model_keep.eval()
        model_strip.eval()

        with torch.no_grad():
            out_keep = model_keep.forward({"graph": _make_graph_batch(batch_size=3)})
            # Re-batch because forward can mutate the graph batch in place.
            out_strip = model_strip.forward({"graph": _make_graph_batch(batch_size=3)})

        assert torch.allclose(out_keep, out_strip, atol=1e-6)


class TestKeepExistingPredictorNested:
    """Nested finetuner: outer flag must reach the intermediate + leaf predictors."""

    @pytest.mark.parametrize("strategy", ["full", "lora"])
    def test_false_strips_entire_chain(self, strategy):
        model = _make_nested_finetuner(
            "classic", strategy, keep_existing_predictor=False
        )
        predictors = _collect_predictors(model)
        # Chain has two levels: intermediate nested Finetuner + leaf classic model.
        assert len(predictors) == 2
        for p in predictors:
            assert isinstance(p, PassthroughPredictor)
        # Head sees the leaf encoder's fp_dim.
        leaf = _resolve_leaf_encoder(model.pretrain)
        assert model.pretrain_output_dim == leaf.fp_dim

    def test_true_leaves_intermediate_predictor_intact(self):
        # Outer flag ``True`` and nested's own flag ``True`` (historic default)
        # must preserve the intermediate MLP head between the two finetuners.
        model = _make_nested_finetuner(
            "classic",
            "full",
            keep_existing_predictor=True,
            nested_keep_existing_predictor=True,
        )
        # Intermediate Finetuner's own .predictor is still an MLP with params.
        intermediate_predictor = model.pretrain.predictor
        assert not isinstance(intermediate_predictor, PassthroughPredictor)
        assert sum(p.numel() for p in intermediate_predictor.parameters()) > 0

    def test_false_full_setup_excludes_stripped_predictor_params(self):
        model = _make_nested_finetuner("classic", "full", keep_existing_predictor=False)
        # PassthroughPredictor has zero parameters at every level, so nothing
        # from the stripped chain can appear in any optimizer group.
        for predictor in _collect_predictors(model):
            assert sum(p.numel() for p in predictor.parameters()) == 0
        # Belt-and-suspenders: the encoder groups only contain leaf-encoder params.
        leaf = _resolve_leaf_encoder(model.pretrain)
        leaf_encoder_ids = {id(p) for p in leaf.parameters()}
        for group in model.pretrain_optimizer.param_groups:
            for p in group["params"]:
                # No stray parameter that isn't part of the leaf encoder.
                assert id(p) in leaf_encoder_ids

    def test_forward_pass_shape_under_false(self):
        model = _make_nested_finetuner("classic", "full", keep_existing_predictor=False)
        model.eval()
        batch = {"graph": _make_graph_batch(batch_size=2)}
        with torch.no_grad():
            out = model.forward(batch)
        assert out.shape == (2, 1)


class TestKeepExistingPredictorCheckpointRoundTrip:
    """Save-with-``False`` must reconstruct the passthrough chain on reload.

    The sklearn wrapper reloads finetuners through the ``_SELF_CONTAINED_SENTINEL``
    skeleton path (:meth:`Finetuner._build_skeleton_from_config`) with the
    saved hparams re-supplied to ``__init__``. This test drives that path
    directly so no filesystem checkpoint artifact is required.
    """

    def test_flat_classic_round_trip_predictions_identical(self):
        model = _make_flat_finetuner("classic", "full", keep_existing_predictor=False)
        model.eval()

        batch = {"graph": _make_graph_batch(batch_size=2)}
        with torch.no_grad():
            before = model.forward(batch)

        # Rebuild via the skeleton path (what load_from_checkpoint drives
        # for a self-contained finetuner) and copy weights across.
        rebuilt = _make_flat_finetuner("classic", "full", keep_existing_predictor=False)
        rebuilt.load_state_dict(model.state_dict())
        rebuilt.eval()

        # Passthrough chain survived the round trip.
        assert isinstance(rebuilt.pretrain.predictor, PassthroughPredictor)
        assert rebuilt.hparams["keep_existing_predictor"] is False

        batch = {"graph": _make_graph_batch(batch_size=2)}
        with torch.no_grad():
            after = rebuilt.forward(batch)
        assert torch.allclose(before, after, atol=1e-6)

    def test_nested_round_trip_predictions_identical(self):
        model = _make_nested_finetuner("classic", "full", keep_existing_predictor=False)
        model.eval()

        batch = {"graph": _make_graph_batch(batch_size=2)}
        with torch.no_grad():
            before = model.forward(batch)

        rebuilt = _make_nested_finetuner(
            "classic", "full", keep_existing_predictor=False
        )
        rebuilt.load_state_dict(model.state_dict())
        rebuilt.eval()

        for p in _collect_predictors(rebuilt):
            assert isinstance(p, PassthroughPredictor)

        batch = {"graph": _make_graph_batch(batch_size=2)}
        with torch.no_grad():
            after = rebuilt.forward(batch)
        assert torch.allclose(before, after, atol=1e-6)
