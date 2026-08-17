"""Regression tests for nested-Finetuner encoder resolution.

Covers issue #67 (LoRA crash on phase-2 artifacts) and the parallel silent
degradation of full-strategy layer-wise LR decay when ``self.pretrain`` is a
nested :class:`Finetuner`. Uses the ``_SELF_CONTAINED_SENTINEL`` skeleton path
so all fixtures live in memory.
"""

import torch.nn as nn

from matcha.torch.models.finetuning.finetuner import (
    Finetuner,
    _SELF_CONTAINED_SENTINEL,
    _resolve_leaf_encoder,
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
):
    """Build a Finetuner whose ``pretrain`` is itself a Finetuner over ``leaf_origin``."""
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

    outer_config = {
        "origin_type": "finetuner",
        "pretrain_params": outer_pretrain_params,
        "source_class": None,
        "nested_pretrain_config": inner_config,
        "nested_hparams": NESTED_HPARAMS,
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
        _pretrain_config=outer_config,
    )


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
