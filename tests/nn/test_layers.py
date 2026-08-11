"""Tests for matcha.nn.layers – LayerRegistry and layer modules."""

import pytest
import torch
from torch import nn

from matcha.nn.layers import (
    LayerRegistry,
    AdaRMSN,
    LnBnDr,
    MultiLn,
    MultiBatchNorm,
    MultiMLP,
    SpatialEncoder,
    SpatialEncoder3d,
    BiasedMultiHeadAttention,
)


# ===================================================================
# Registry completeness
# ===================================================================


class TestLayerRegistryKeys:
    EXPECTED_KEYS = [
        "adarmsn",
        "batch",
        "layer",
        "instance",
        "lnbndr",
        "multiln",
        "multibatch",
        "multimlp",
        "spatial_encoder",
        "spatial_encoder_3d",
        "biased_mha",
    ]

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_key_registered(self, key):
        assert key in LayerRegistry, f"'{key}' not found in LayerRegistry"


# ===================================================================
# AdaRMSN
# ===================================================================


class TestAdaRMSN:
    def test_output_shape(self):
        norm = AdaRMSN(dim=32)
        x = torch.randn(8, 32)
        out = norm(x)
        assert out.shape == (8, 32)

    def test_output_3d(self):
        norm = AdaRMSN(dim=16)
        x = torch.randn(4, 10, 16)
        out = norm(x)
        assert out.shape == (4, 10, 16)

    def test_output_is_finite(self):
        norm = AdaRMSN(dim=32)
        x = torch.randn(8, 32)
        out = norm(x)
        assert torch.isfinite(out).all()

    def test_learnable_params(self):
        norm = AdaRMSN(dim=16)
        param_names = {n for n, _ in norm.named_parameters()}
        assert "beta" in param_names
        assert "alpha" in param_names

    def test_grad_flows(self):
        norm = AdaRMSN(dim=32)
        x = torch.randn(8, 32, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None


# ===================================================================
# Standard norms – only check registry wiring
# ===================================================================


class TestStandardNorms:
    @pytest.mark.parametrize(
        "key,expected_parent",
        [
            ("batch", nn.BatchNorm1d),
            ("layer", nn.LayerNorm),
            ("instance", nn.InstanceNorm1d),
        ],
    )
    def test_is_subclass(self, key, expected_parent):
        norm_cls = LayerRegistry[key]
        assert issubclass(norm_cls, expected_parent)


# ===================================================================
# LnBnDr
# ===================================================================


class TestLnBnDr:
    def test_output_shape(self):
        layer = LnBnDr(
            input_dim=32,
            output_dim=16,
            dropout=0.1,
            activation="relu",
            norm="layer",
        )
        x = torch.randn(8, 32)
        out = layer(x)
        assert out.shape == (8, 16)

    def test_output_shape_no_activation(self):
        layer = LnBnDr(
            input_dim=32,
            output_dim=16,
            dropout=0.0,
            activation=None,
            norm="layer",
        )
        x = torch.randn(8, 32)
        out = layer(x)
        assert out.shape == (8, 16)

    def test_output_shape_no_norm(self):
        layer = LnBnDr(
            input_dim=32,
            output_dim=16,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        x = torch.randn(8, 32)
        out = layer(x)
        assert out.shape == (8, 16)

    def test_in_out_features(self):
        layer = LnBnDr(
            input_dim=64,
            output_dim=32,
            dropout=0.0,
            activation="relu",
            norm="layer",
        )
        assert layer.in_features == 64
        assert layer.out_features == 32

    def test_grad_flows(self):
        layer = LnBnDr(
            input_dim=32,
            output_dim=16,
            dropout=0.0,
            activation="relu",
            norm="layer",
        )
        x = torch.randn(8, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize("activation", ["relu", "gelu", "swish", "tanh"])
    def test_various_activations(self, activation):
        layer = LnBnDr(
            input_dim=32,
            output_dim=16,
            dropout=0.0,
            activation=activation,
            norm="layer",
        )
        x = torch.randn(8, 32)
        out = layer(x)
        assert out.shape == (8, 16)


# ===================================================================
# MultiLn
# ===================================================================


class TestMultiLn:
    def test_output_shape(self):
        layer = MultiLn(in_dim=32, out_dim=16, num_parallel=4)
        # Expected input: (num_parallel, batch_size, in_dim)
        x = torch.randn(4, 8, 32)
        out = layer(x)
        assert out.shape == (4, 8, 16)

    def test_with_bias(self):
        layer = MultiLn(in_dim=32, out_dim=16, num_parallel=4, bias=True)
        assert layer.bias is not None

    def test_without_bias(self):
        layer = MultiLn(in_dim=32, out_dim=16, num_parallel=4, bias=False)
        assert layer.bias is None

    def test_reset_parameters(self):
        layer = MultiLn(in_dim=32, out_dim=16, num_parallel=4)
        # After reset, bias should be zeros
        if layer.bias is not None:
            assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_grad_flows(self):
        layer = MultiLn(in_dim=32, out_dim=16, num_parallel=4)
        x = torch.randn(4, 8, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


# ===================================================================
# MultiBatchNorm
# ===================================================================


class TestMultiBatchNorm:
    def test_output_shape(self):
        norm = MultiBatchNorm(num_features=16, num_parallel=4)
        norm.train()
        # Input: (num_parallel, batch_size, num_features)
        x = torch.randn(4, 8, 16)
        out = norm(x)
        assert out.shape == (4, 8, 16)

    def test_eval_uses_running_stats(self):
        norm = MultiBatchNorm(num_features=16, num_parallel=4)
        # Train for a few steps
        norm.train()
        for _ in range(5):
            x = torch.randn(4, 8, 16)
            norm(x)
        # Switch to eval
        norm.eval()
        x_eval = torch.randn(4, 8, 16)
        out = norm(x_eval)
        assert out.shape == (4, 8, 16)
        assert torch.isfinite(out).all()

    def test_running_stats_updated_during_training(self):
        norm = MultiBatchNorm(num_features=16, num_parallel=4)
        norm.train()
        initial_mean = norm.running_mean.clone()
        x = torch.randn(4, 8, 16) + 5.0  # shift mean
        norm(x)
        assert not torch.allclose(norm.running_mean, initial_mean)


# ===================================================================
# MultiMLP
# ===================================================================


class TestMultiMLP:
    def test_output_shape(self):
        mlp = MultiMLP(
            input_dim=32,
            dims=[16, 8],
            num_parallel=3,
            dropout=0.0,
            activation="relu",
            norm="multibatch",
        )
        mlp.train()
        # Input: (num_parallel, batch_size, input_dim)
        x = torch.randn(3, 8, 32)
        out = mlp(x)
        # Output should be (batch_size, num_parallel) since final layer is 1-dim per head
        assert out.shape == (8, 3)

    def test_grad_flows(self):
        mlp = MultiMLP(
            input_dim=32,
            dims=[16],
            num_parallel=2,
            dropout=0.0,
            activation="relu",
            norm="multibatch",
        )
        mlp.train()
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None


# ===================================================================
# SpatialEncoder
# ===================================================================


class TestSpatialEncoder:
    def test_output_shape(self):
        enc = SpatialEncoder(max_dist=5, num_heads=4)
        spd = torch.randint(0, 6, (2, 10, 10))
        out = enc(spd)
        assert out.shape == (2, 10, 10, 4)

    def test_handles_negative_distances(self):
        """Negative distances (unreachable) should be clamped to max_dist+1."""
        enc = SpatialEncoder(max_dist=5, num_heads=4)
        spd = torch.tensor([[[-1, 2], [3, -1]]])
        out = enc(spd)
        assert out.shape == (1, 2, 2, 4)
        assert torch.isfinite(out).all()

    def test_handles_large_distances(self):
        enc = SpatialEncoder(max_dist=5, num_heads=4)
        spd = torch.tensor([[[100, 2], [3, 50]]])
        out = enc(spd)
        assert torch.isfinite(out).all()


# ===================================================================
# SpatialEncoder3d
# ===================================================================


class TestSpatialEncoder3d:
    def test_output_shape(self):
        enc = SpatialEncoder3d(num_kernels=8, num_heads=4, atom_feat_dim=16)
        coord = torch.randn(2, 10, 3)
        atom_feats = torch.randn(2, 10, 16)
        out = enc(coord, atom_feats)
        assert out.shape == (2, 10, 10, 4)

    def test_output_is_finite(self):
        enc = SpatialEncoder3d(num_kernels=8, num_heads=4, atom_feat_dim=16)
        coord = torch.randn(2, 5, 3)
        atom_feats = torch.randn(2, 5, 16)
        out = enc(coord, atom_feats)
        assert torch.isfinite(out).all()

    def test_grad_flows_through_coord(self):
        enc = SpatialEncoder3d(num_kernels=8, num_heads=4, atom_feat_dim=16)
        coord = torch.randn(2, 5, 3, requires_grad=True)
        atom_feats = torch.randn(2, 5, 16)
        out = enc(coord, atom_feats)
        out.sum().backward()
        assert coord.grad is not None

    def test_spatial3d_distances_produce_distinct_bias(self):
        enc = SpatialEncoder3d(num_kernels=8, num_heads=4, atom_feat_dim=16)
        # Pin kernel centres/widths so distances 1.0 and 3.0 land in distinct kernel regions.
        # softplus(-0.4328) ≈ 0.5 — matches the original std=0.5 used before reparameterization.
        enc.means.data = torch.linspace(0, 4, 8)
        enc.raw_stds.data = torch.full((8,), -0.4328)
        # atom_feats = 1/D so gamma_proj(feats) = sum(1 * 1/16)*16 = 1 per atom
        # → gamma_ij = 2, scaled distances: 2*0.5=1.0 vs 2*1.5=3.0
        atom_feats = torch.ones(1, 2, 16) / 16

        coord_close = torch.zeros(1, 2, 3)
        coord_close[0, 1, 0] = 0.5
        out_close = enc(coord_close, atom_feats)

        coord_far = torch.zeros(1, 2, 3)
        coord_far[0, 1, 0] = 1.5
        out_far = enc(coord_far, atom_feats)

        assert not torch.allclose(out_close[0, 0, 1], out_far[0, 0, 1])

    def test_spatial3d_gradient_flows_through_atom_feats(self):
        enc = SpatialEncoder3d(num_kernels=8, num_heads=4, atom_feat_dim=16)
        coord = torch.randn(2, 5, 3)
        atom_feats = torch.randn(2, 5, 16, requires_grad=True)
        out = enc(coord, atom_feats)
        out.sum().backward()
        assert atom_feats.grad is not None

    def test_spatial3d_init_beta_is_zero(self):
        enc = SpatialEncoder3d(num_kernels=8, num_heads=4, atom_feat_dim=16)
        atom_feats = torch.randn(1, 3, 16)
        beta = enc.beta_proj(atom_feats)
        assert torch.allclose(beta, torch.zeros_like(beta))
        gamma = enc.gamma_proj(atom_feats)
        assert not torch.allclose(gamma, torch.zeros_like(gamma))


# ===================================================================
# BiasedMultiHeadAttention
# ===================================================================


class TestBiasedMultiHeadAttention:
    def test_output_shape(self):
        attn = BiasedMultiHeadAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_with_attn_bias(self):
        attn = BiasedMultiHeadAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        bias = torch.randn(2, 10, 10, 4)
        out = attn(x, attn_bias=bias)
        assert out.shape == (2, 10, 32)

    def test_with_attn_mask(self):
        attn = BiasedMultiHeadAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 5:] = False  # mask out last 5 positions for first batch
        out = attn(x, attn_mask=mask)
        assert out.shape == (2, 10, 32)
        assert torch.isfinite(out).all()

    def test_with_bias_and_mask(self):
        attn = BiasedMultiHeadAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        bias = torch.randn(2, 10, 10, 4)
        mask = torch.ones(2, 10, dtype=torch.bool)
        out = attn(x, attn_bias=bias, attn_mask=mask)
        assert out.shape == (2, 10, 32)

    def test_embed_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            BiasedMultiHeadAttention(embed_dim=33, num_heads=4)

    def test_grad_flows(self):
        attn = BiasedMultiHeadAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


# ===================================================================
# from_dense_batch utility
# ===================================================================


class TestFromDenseBatch:
    def test_roundtrip(self):
        from matcha.nn.layers import from_dense_batch

        batch = torch.tensor([0, 0, 0, 1, 1])
        # Dense: (2, 3, feat_size)
        dense = torch.randn(2, 3, 8)
        # Fill appropriately
        dense[1, 2, :] = 0  # graph 1 only has 2 nodes
        reconstructed = from_dense_batch(dense, batch)
        assert reconstructed.shape == (5, 8)
