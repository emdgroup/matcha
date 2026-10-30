"""Test HPOManager through the sklearn API.

Model: GTRegressor (graph), ChempropRegressor (chemprop)
Exercises: tune with minimal budget (architecture_search_budget=2,
    optimizer_search_budget=2). Verifies that tuning completes and updates
    the model parameters.

A custom architecture_grid is provided so that enc_atom_hidden_dim is
always divisible by enc_num_heads (the default grid can randomly sample
incompatible combinations that trigger an AssertionError inside GTConv).
"""

import pytest

from matcha.sklearn.clm import (
    CNNClassifier,
    CNNRegressor,
    RNNClassifier,
    RNNRegressor,
    RoFormerClassifier,
    RoFormerRegressor,
)
from matcha.sklearn.graph import ChempropRegressor, GTRegressor
from matcha.sklearn.managers import HPOManager


@pytest.fixture()
def model_kwargs():
    return dict(
        enc_atom_hidden_dim=32,
        enc_num_layers=1,
        enc_num_heads=4,
        enc_expansion_k=2,
        pred_hidden_dims=[32],
        rwse_k=0,
        laplacian_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
        num_virtual_nodes=0,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


# A safe architecture grid: hidden_dim is always a multiple of every
# candidate num_heads value (4 and 8 both divide 64, 128, 256).
_SAFE_ARCHITECTURE_GRID = {
    "enc_atom_hidden_dim": ["choice", [64, 128]],
    "enc_num_heads": ["choice", [4, 8]],
    "enc_num_layers": ["int", {"low": 1, "high": 2}],
    "enc_readout": ["choice", ["attentive", "vpa"]],
    "enc_jk": ["choice", ["last", "sum"]],
    "pred_hidden_dims": ["choice", [[32], [64]]],
    "enc_dropout": ["float", {"low": 0.0, "high": 0.2}],
    "pred_dropout": ["float", {"low": 0.0, "high": 0.2}],
    "enc_activation": ["choice", ["gelu"]],
    "pred_activation": ["choice", ["gelu"]],
}


class TestHPOManagerInit:
    """Tests for initial state of HPOManager."""

    def test_params_is_none_by_default(self):
        mgr = HPOManager()
        assert mgr.params is None


class TestHPOManagerTune:
    """Tests for tune through the sklearn API.

    Uses a very small search budget so the test finishes quickly.
    """

    def test_tune_completes(self, mol_list, regression_y, model_kwargs):
        model = GTRegressor(**model_kwargs)

        # Split molecules into train / val portions
        split_idx = int(len(mol_list) * 0.8)
        train_mols = mol_list[:split_idx]
        val_mols = mol_list[split_idx:]
        train_y = regression_y[:split_idx]
        val_y = regression_y[split_idx:]

        # Featurize each split as a proper StackDataset
        train_set = model.transform(train_mols, train_y, is_training=True)
        val_set = model.transform(val_mols, val_y, is_training=False)

        arc_study, opt_study = model.tune(
            train_set=train_set,
            val_set=[val_set, val_set],
            architecture_search_budget=2,
            architecture_grid=_SAFE_ARCHITECTURE_GRID,
            optimizer_search_budget=2,
        )
        assert arc_study is not None
        assert opt_study is not None

    def test_tune_populates_hpo_params(self, mol_list, regression_y, model_kwargs):
        model = GTRegressor(**model_kwargs)

        split_idx = int(len(mol_list) * 0.8)
        train_mols = mol_list[:split_idx]
        val_mols = mol_list[split_idx:]
        train_y = regression_y[:split_idx]
        val_y = regression_y[split_idx:]

        train_set = model.transform(train_mols, train_y, is_training=True)
        val_set = model.transform(val_mols, val_y, is_training=False)

        model.tune(
            train_set=train_set,
            val_set=[val_set, val_set],
            architecture_search_budget=2,
            architecture_grid=_SAFE_ARCHITECTURE_GRID,
            optimizer_search_budget=2,
        )
        assert model._hpo_manager.params is not None


# =========================================================================
# Chemprop fixtures and architecture grid
# =========================================================================


@pytest.fixture()
def chemprop_model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dim=32,
        pred_num_layers=1,
        feature_list=None,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


_CHEMPROP_ARCHITECTURE_GRID = {
    "enc_atom_hidden_dim": ["choice", [64, 128]],
    "enc_num_layers": ["int", {"low": 1, "high": 2}],
    "enc_dropout": ["float", {"low": 0.0, "high": 0.2}],
    "enc_activation": ["choice", ["relu"]],
    "pred_hidden_dim": ["choice", [64, 128]],
    "pred_num_layers": ["int", {"low": 1, "high": 2}],
    "pred_dropout": ["float", {"low": 0.0, "high": 0.2}],
    "pred_activation": ["choice", ["relu"]],
}


class TestChempropHPOManagerTune:
    """Tests for tune through the sklearn API with ChempropRegressor.

    Uses a very small search budget so the test finishes quickly.
    """

    def test_tune_completes(self, mol_list, regression_y, chemprop_model_kwargs):
        model = ChempropRegressor(**chemprop_model_kwargs)

        split_idx = int(len(mol_list) * 0.8)
        train_mols = mol_list[:split_idx]
        val_mols = mol_list[split_idx:]
        train_y = regression_y[:split_idx]
        val_y = regression_y[split_idx:]

        train_set = model.transform(train_mols, train_y, is_training=True)
        val_set = model.transform(val_mols, val_y, is_training=False)

        arc_study, opt_study = model.tune(
            train_set=train_set,
            val_set=[val_set, val_set],
            architecture_search_budget=2,
            architecture_grid=_CHEMPROP_ARCHITECTURE_GRID,
            optimizer_search_budget=2,
        )
        assert arc_study is not None
        assert opt_study is not None

    def test_tune_populates_hpo_params(
        self, mol_list, regression_y, chemprop_model_kwargs
    ):
        model = ChempropRegressor(**chemprop_model_kwargs)

        split_idx = int(len(mol_list) * 0.8)
        train_mols = mol_list[:split_idx]
        val_mols = mol_list[split_idx:]
        train_y = regression_y[:split_idx]
        val_y = regression_y[split_idx:]

        train_set = model.transform(train_mols, train_y, is_training=True)
        val_set = model.transform(val_mols, val_y, is_training=False)

        model.tune(
            train_set=train_set,
            val_set=[val_set, val_set],
            architecture_search_budget=2,
            architecture_grid=_CHEMPROP_ARCHITECTURE_GRID,
            optimizer_search_budget=2,
        )
        assert model._hpo_manager.params is not None


# =========================================================================
# CLM fixtures and architecture grids
# =========================================================================

_CLM_SHARED = dict(
    num_epochs=1,
    batch_size=32,
    accelerator="cpu",
    devices=1,
    early_stopping=False,
    stochastic_weight_averaging=False,
    num_augmentations=1,
    max_length=100,
)

# Safe architecture grids: enc_hidden_dim choices (32, 64) are both divisible
# by every candidate enc_num_heads value (4, 8), so parse_num_heads never
# needs to fall back and no trial ever sees an incompatible combination.
_CLM_CNN_ARCHITECTURE_GRID = {
    "enc_kernel_dims": ["choice", [[3, 5], [3, 5, 7]]],
    "enc_hidden_dim": ["choice", [32, 64]],
    "enc_num_heads": ["choice", [4, 8]],
}

_CLM_RNN_ARCHITECTURE_GRID = {
    "enc_num_layers": ["int", {"low": 1, "high": 2}],
    "enc_embedding_dim": ["choice", [32, 64]],
    "enc_hidden_dim": ["choice", [32, 64]],
    "enc_num_heads": ["choice", [4, 8]],
}

_CLM_ROFORMER_ARCHITECTURE_GRID = {
    "enc_expansion_dim": ["choice", [64, 128]],
    "enc_num_layers": ["int", {"low": 1, "high": 2}],
    "enc_hidden_dim": ["choice", [32, 64]],
    "enc_num_heads": ["choice", [4, 8]],
}

_CLM_ARCH_KWARGS: dict[type, dict] = {
    CNNRegressor: {
        **_CLM_SHARED,
        "enc_hidden_dim": 32,
        "enc_kernel_dims": [3, 5],
        "enc_num_heads": 4,
        "pred_hidden_dims": [32],
    },
    CNNClassifier: {
        **_CLM_SHARED,
        "enc_hidden_dim": 32,
        "enc_kernel_dims": [3, 5],
        "enc_num_heads": 4,
        "pred_hidden_dims": [32],
    },
    RNNRegressor: {
        **_CLM_SHARED,
        "enc_num_layers": 1,
        "enc_embedding_dim": 32,
        "enc_hidden_dim": 32,
        "enc_num_heads": 4,
        "enc_bidirectional": False,
        "pred_hidden_dims": [32],
    },
    RNNClassifier: {
        **_CLM_SHARED,
        "enc_num_layers": 1,
        "enc_embedding_dim": 32,
        "enc_hidden_dim": 32,
        "enc_num_heads": 4,
        "enc_bidirectional": False,
        "pred_hidden_dims": [32],
    },
    RoFormerRegressor: {
        **_CLM_SHARED,
        "enc_hidden_dim": 32,
        "enc_expansion_dim": 64,
        "enc_num_heads": 4,
        "enc_num_layers": 1,
        "pred_hidden_dims": [32],
    },
    RoFormerClassifier: {
        **_CLM_SHARED,
        "enc_hidden_dim": 32,
        "enc_expansion_dim": 64,
        "enc_num_heads": 4,
        "enc_num_layers": 1,
        "pred_hidden_dims": [32],
    },
}

_CLM_ARCHITECTURE_GRIDS: dict[type, dict] = {
    CNNRegressor: _CLM_CNN_ARCHITECTURE_GRID,
    CNNClassifier: _CLM_CNN_ARCHITECTURE_GRID,
    RNNRegressor: _CLM_RNN_ARCHITECTURE_GRID,
    RNNClassifier: _CLM_RNN_ARCHITECTURE_GRID,
    RoFormerRegressor: _CLM_ROFORMER_ARCHITECTURE_GRID,
    RoFormerClassifier: _CLM_ROFORMER_ARCHITECTURE_GRID,
}

_CLM_CLASSES = [
    CNNRegressor,
    CNNClassifier,
    RNNRegressor,
    RNNClassifier,
    RoFormerRegressor,
    RoFormerClassifier,
]


class TestCLMHPOManagerTune:
    """Tests for tune() across all 6 CLM variants.

    Verifies both that tuning completes without crashing and that
    enc_num_characters is correctly patched to the fitted vocabulary size.
    """

    @pytest.mark.parametrize("clm_cls", _CLM_CLASSES, ids=lambda c: c.__name__)
    def test_clm_tune_completes(
        self, clm_cls, mol_list, regression_y, classification_y
    ):
        y = regression_y if "Regressor" in clm_cls.__name__ else classification_y
        model = clm_cls(**_CLM_ARCH_KWARGS[clm_cls])

        split_idx = int(len(mol_list) * 0.8)
        train_set = model.transform(
            mol_list[:split_idx], y[:split_idx], is_training=True
        )
        val_set = model.transform(
            mol_list[split_idx:], y[split_idx:], is_training=False
        )

        arc_study, opt_study = model.tune(
            train_set=train_set,
            val_set=[val_set, val_set],
            architecture_search_budget=2,
            architecture_grid=_CLM_ARCHITECTURE_GRIDS[clm_cls],
            optimizer_search_budget=2,
        )
        assert arc_study is not None
        assert opt_study is not None

    @pytest.mark.parametrize("clm_cls", _CLM_CLASSES, ids=lambda c: c.__name__)
    def test_clm_tune_patches_enc_num_characters(
        self, clm_cls, mol_list, regression_y, classification_y
    ):
        y = regression_y if "Regressor" in clm_cls.__name__ else classification_y
        model = clm_cls(**_CLM_ARCH_KWARGS[clm_cls])

        split_idx = int(len(mol_list) * 0.8)
        train_set = model.transform(
            mol_list[:split_idx], y[:split_idx], is_training=True
        )
        val_set = model.transform(
            mol_list[split_idx:], y[split_idx:], is_training=False
        )

        model.tune(
            train_set=train_set,
            val_set=[val_set, val_set],
            architecture_search_budget=2,
            architecture_grid=_CLM_ARCHITECTURE_GRIDS[clm_cls],
            optimizer_search_budget=2,
        )

        expected = model.datamodule.params.num_tokens
        assert model._model.params.enc_num_characters > 4
        assert model._model.params.enc_num_characters == expected
