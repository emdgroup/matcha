"""Tests for sentinel-based self-contained Finetuner loading.

Verifies that `Finetuner.__init__` and `ChempropFinetuner.__init__` with
`path_to_pretrained="__self_contained__"` and a `_pretrain_config` dict produce
a valid module graph with correct dimensions, without any filesystem access.
"""

import pytest
import torch

from matcha.torch.models.finetuning.finetuner import Finetuner, _SELF_CONTAINED_SENTINEL
from matcha.torch.models.finetuning.chemprop_finetuner import ChempropFinetuner


# =========================================================================
# Minimal pretrain_params for different origin types
# =========================================================================

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


class TestSelfContainedClassicOrigin:
    """Test skeleton building from a classic (GIN) origin type."""

    def _make_finetuner(self):
        config = {
            "origin_type": "classic",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
        }
        return Finetuner(
            architecture="ginmodel",
            path_to_pretrained=_SELF_CONTAINED_SENTINEL,
            pred_hidden_dims=[32],
            num_endpoints=1,
            dropout=0.0,
            activation="relu",
            optimizer_args={"lr": 1e-4},
            scheduler_args={"min_lr": 1e-6, "total_steps": 50},
            _pretrain_config=config,
        )

    def test_creates_valid_module_graph(self):
        model = self._make_finetuner()
        assert hasattr(model, "pretrain")
        assert hasattr(model, "predictor")
        assert model.pretrain is not None
        assert model.predictor is not None

    def test_pretrain_has_encoder(self):
        model = self._make_finetuner()
        assert hasattr(model.pretrain, "encoder")
        assert model.pretrain.encoder is not None

    def test_latent_dim_matches(self):
        model = self._make_finetuner()
        # The predictor input_dim should match pretrain output
        assert model.pretrain_output_dim == model.pretrain.latent_dim

    def test_forward_shape(self):
        model = self._make_finetuner()
        # Verify the predictor can accept input of pretrain_output_dim
        dummy_input = torch.randn(2, model.pretrain_output_dim)
        output = model.predictor(dummy_input)
        assert output.shape == (2, 1)  # num_endpoints=1


class TestSelfContainedPretrainingOrigin:
    """Test skeleton building from a pretraining origin type."""

    def _make_finetuner(self):
        config = {
            "origin_type": "pretraining",
            "pretrain_params": GIN_PRETRAINING_PARAMS,
            "source_class": "GINPretraining",
        }
        return Finetuner(
            architecture="ginmodel",
            path_to_pretrained=_SELF_CONTAINED_SENTINEL,
            pred_hidden_dims=[32],
            num_endpoints=1,
            dropout=0.0,
            activation="relu",
            optimizer_args={"lr": 1e-4},
            scheduler_args={"min_lr": 1e-6, "total_steps": 50},
            _pretrain_config=config,
        )

    def test_creates_encoder_wrapper(self):
        from matcha.torch.models.finetuning.pretrained_encoder_wrapper import (
            PretrainedEncoderWrapper,
        )

        model = self._make_finetuner()
        assert isinstance(model.pretrain, PretrainedEncoderWrapper)

    def test_encoder_type_is_graph(self):
        model = self._make_finetuner()
        assert model.pretrain._encoder_type == "graph"

    def test_latent_dim_matches(self):
        model = self._make_finetuner()
        assert model.pretrain_output_dim == model.pretrain.latent_dim


class TestSelfContainedNestedFinetuner:
    """Test skeleton building for nested finetuner (A → B → C scenario)."""

    def _make_nested_finetuner(self):
        # Inner config: B was finetuned from a classic GIN model (A)
        inner_config = {
            "origin_type": "classic",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
        }
        # Outer config: C was finetuned from B (a finetuner)
        outer_config = {
            "origin_type": "finetuner",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
            "nested_pretrain_config": inner_config,
            "nested_hparams": {
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
            },
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
            _pretrain_config=outer_config,
        )

    def test_nested_finetuner_is_finetuner(self):
        model = self._make_nested_finetuner()
        assert isinstance(model.pretrain, Finetuner)

    def test_nested_pretrain_is_classic(self):
        model = self._make_nested_finetuner()
        # model.pretrain is B (Finetuner), model.pretrain.pretrain is A (classic GIN)
        inner_pretrain = model.pretrain.pretrain
        assert hasattr(inner_pretrain, "encoder")

    def test_nested_latent_dims_consistent(self):
        model = self._make_nested_finetuner()
        # Outer predictor input should match inner finetuner's latent_dim
        assert model.pretrain_output_dim == model.pretrain.latent_dim


class TestBuildPretrainConfig:
    """Test that _build_pretrain_config captures correct metadata."""

    def test_classic_origin_config(self):
        config = {
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
            _pretrain_config=config,
        )
        result = model._build_pretrain_config()
        assert result["origin_type"] == "classic"
        assert result["pretrain_params"] == GIN_PRETRAIN_PARAMS

    def test_nested_finetuner_config_recursive(self):
        inner_config = {
            "origin_type": "classic",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
        }
        outer_config = {
            "origin_type": "finetuner",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
            "nested_pretrain_config": inner_config,
            "nested_hparams": {
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
            },
        }
        model = Finetuner(
            architecture="finetunermodel",
            path_to_pretrained=_SELF_CONTAINED_SENTINEL,
            pred_hidden_dims=[32],
            num_endpoints=1,
            dropout=0.0,
            activation="relu",
            optimizer_args={"lr": 1e-4},
            scheduler_args={"min_lr": 1e-6, "total_steps": 50},
            _pretrain_config=outer_config,
        )
        result = model._build_pretrain_config()
        assert result["origin_type"] == "finetuner"
        assert "nested_pretrain_config" in result
        assert result["nested_pretrain_config"]["origin_type"] == "classic"


class TestSentinelGuardRails:
    """Test error handling for invalid sentinel usage."""

    def test_sentinel_without_config_raises(self):
        with pytest.raises(ValueError, match="Cannot build skeleton"):
            Finetuner(
                architecture="ginmodel",
                path_to_pretrained=_SELF_CONTAINED_SENTINEL,
                pred_hidden_dims=[32],
                num_endpoints=1,
                optimizer_args={"lr": 1e-4},
                scheduler_args={"min_lr": 1e-6, "total_steps": 50},
                _pretrain_config=None,
            )

    def test_invalid_origin_type_raises(self):
        config = {
            "origin_type": "unknown",
            "pretrain_params": GIN_PRETRAIN_PARAMS,
            "source_class": None,
        }
        with pytest.raises(ValueError, match="Unknown origin_type"):
            Finetuner(
                architecture="ginmodel",
                path_to_pretrained=_SELF_CONTAINED_SENTINEL,
                pred_hidden_dims=[32],
                num_endpoints=1,
                optimizer_args={"lr": 1e-4},
                scheduler_args={"min_lr": 1e-6, "total_steps": 50},
                _pretrain_config=config,
            )


# =========================================================================
# ChempropFinetuner self-contained tests (issue #396, stage 3)
# =========================================================================

CHEMPROP_PRETRAIN_PARAMS = {
    "enc_atom_hidden_dim": 32,
    "enc_num_layers": 1,
    "enc_dropout": 0.0,
    "enc_activation": "relu",
    "enc_readout": "norm",
    "additional_mol_features_dim": 0,
    "pred_hidden_dim": 32,
    "pred_num_layers": 1,
    "pred_dropout": 0.0,
    "pred_activation": "relu",
    "num_endpoints": 1,
    "loss_fn": "mse",
    "optimizer": "chemprop",
    "optimizer_args": {"lr": 1e-3},
    "scheduler": "chemprop",
    "scheduler_args": {"warmup_epochs": 5, "max_lr": 1e-2, "final_lr": 1e-5},
}


class TestSelfContainedChempropOrigin:
    """Test skeleton building from a chemprop origin type."""

    def _make_chemprop_finetuner(self, num_endpoints=1):
        config = {
            "origin_type": "chemprop",
            "pretrain_params": CHEMPROP_PRETRAIN_PARAMS,
        }
        return ChempropFinetuner(
            path_to_pretrained=_SELF_CONTAINED_SENTINEL,
            pred_hidden_dim=32,
            pred_num_layers=1,
            pred_dropout=0.0,
            pred_activation="relu",
            num_endpoints=num_endpoints,
            loss_fn="mse",
            optimizer="chemprop",
            optimizer_args={"lr": 1e-5},
            scheduler_args={"warmup_epochs": 5, "max_lr": 1e-4, "final_lr": 1e-5},
            _pretrain_config=config,
        )

    def test_creates_valid_module_graph(self):
        """Sentinel + config produces a ChempropFinetuner with message_passing and predictor."""
        model = self._make_chemprop_finetuner()
        assert hasattr(model, "message_passing")
        assert hasattr(model, "predictor")
        assert model.message_passing is not None
        assert model.predictor is not None

    def test_forward_shape(self):
        """Dummy input through predictor gives correct output shape."""
        model = self._make_chemprop_finetuner(num_endpoints=3)
        # The predictor FFN input_dim should match message_passing output_dim
        input_dim = model.predictor.ffn.input_dim
        dummy_input = torch.randn(4, input_dim)
        output = model.predictor(dummy_input)
        assert output.shape == (4, 3)

    def test_build_pretrain_config(self):
        """_build_pretrain_config() returns expected dict with origin_type 'chemprop'."""
        model = self._make_chemprop_finetuner()
        result = model._build_pretrain_config()
        assert result["origin_type"] == "chemprop"
        assert result["pretrain_params"] == CHEMPROP_PRETRAIN_PARAMS

    def test_sentinel_without_config_raises(self):
        """ChempropFinetuner with sentinel but no config raises ValueError."""
        with pytest.raises(ValueError, match="Cannot build skeleton"):
            ChempropFinetuner(
                path_to_pretrained=_SELF_CONTAINED_SENTINEL,
                _pretrain_config=None,
            )
