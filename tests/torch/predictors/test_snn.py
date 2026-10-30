"""Unit tests for the SNN predictor and its BatchEnsembleLinear primitive."""

import math

import pytest
import torch
import torch.nn as nn

from matcha.torch.predictors.snn import SNN, BatchEnsembleLinear


# ===================================================================
# BatchEnsembleLinear
# ===================================================================


class TestBatchEnsembleLinear:
    @pytest.mark.parametrize(
        "batch, in_features, out_features, num_parallel",
        [
            (4, 8, 16, 3),
            (1, 5, 5, 2),
            (16, 32, 4, 8),
        ],
    )
    def test_forward_shape_2d_input(
        self, batch, in_features, out_features, num_parallel
    ):
        layer = BatchEnsembleLinear(in_features, out_features, num_parallel)
        x = torch.randn(batch, in_features)
        out = layer(x)
        assert out.shape == (num_parallel, batch, out_features)

    @pytest.mark.parametrize(
        "batch, in_features, out_features, num_parallel",
        [
            (4, 8, 16, 3),
            (2, 5, 5, 2),
            (12, 16, 8, 4),
        ],
    )
    def test_forward_shape_3d_input(
        self, batch, in_features, out_features, num_parallel
    ):
        layer = BatchEnsembleLinear(in_features, out_features, num_parallel)
        x = torch.randn(num_parallel, batch, in_features)
        out = layer(x)
        assert out.shape == (num_parallel, batch, out_features)

    @pytest.mark.parametrize("bad_dim", [1, 4])
    def test_rejects_bad_dim(self, bad_dim):
        layer = BatchEnsembleLinear(8, 4, 3)
        shape = [8] + [1] * (bad_dim - 1) if bad_dim > 0 else [8]
        x = torch.randn(*shape)
        with pytest.raises(ValueError, match="expects input of shape"):
            layer(x)

    def test_param_count_below_naive_ensemble(self):
        k, in_features, out_features = 8, 64, 32
        layer = BatchEnsembleLinear(in_features, out_features, k)
        total = sum(p.numel() for p in layer.parameters())
        naive = k * in_features * out_features + k * out_features
        assert total < naive

    def test_num_extra_parameters_formula(self):
        k, in_features, out_features = 5, 10, 7
        layer = BatchEnsembleLinear(in_features, out_features, k)
        expected = k * in_features + k * out_features + k * 1 * out_features
        assert layer.num_extra_parameters() == expected

    def test_num_extra_parameters_no_bias(self):
        k, in_features, out_features = 5, 10, 7
        layer = BatchEnsembleLinear(in_features, out_features, k, bias=False)
        expected = k * in_features + k * out_features
        assert layer.num_extra_parameters() == expected

    def test_shared_weight_shape(self):
        layer = BatchEnsembleLinear(8, 4, 3)
        assert layer.weight.shape == (4, 8)

    def test_adapter_shapes(self):
        layer = BatchEnsembleLinear(8, 4, 3)
        assert layer.R.shape == (3, 8)
        assert layer.S.shape == (3, 4)
        assert layer.bias is not None
        assert layer.bias.shape == (3, 1, 4)

    def test_init_first_layer_R_is_rademacher(self):
        torch.manual_seed(0)
        # Wide k * in so both -1 and +1 are effectively certain to be sampled.
        layer = BatchEnsembleLinear(
            in_features=64, out_features=8, num_parallel=8, first_layer=True
        )
        unique = torch.unique(layer.R)
        assert set(unique.tolist()).issubset({-1.0, 1.0})
        assert -1.0 in unique.tolist()
        assert 1.0 in unique.tolist()

    def test_init_hidden_layer_R_is_ones(self):
        layer = BatchEnsembleLinear(
            in_features=8, out_features=4, num_parallel=3, first_layer=False
        )
        assert torch.equal(layer.R, torch.ones_like(layer.R))

    def test_init_S_is_ones_and_bias_zero(self):
        layer = BatchEnsembleLinear(8, 4, 3, first_layer=True)
        assert torch.equal(layer.S, torch.ones_like(layer.S))
        assert layer.bias is not None
        assert torch.equal(layer.bias, torch.zeros_like(layer.bias))

    def test_weight_lecun_normal_std(self):
        torch.manual_seed(0)
        in_features = 1024
        layer = BatchEnsembleLinear(
            in_features=in_features, out_features=64, num_parallel=2
        )
        expected_std = 1.0 / math.sqrt(in_features)
        actual_std = layer.weight.detach().std().item()
        # Generous tolerance — LeCun-normal target is 1/sqrt(fan_in).
        assert actual_std == pytest.approx(expected_std, rel=0.2)

    def test_extra_repr_mentions_first_layer(self):
        layer = BatchEnsembleLinear(8, 4, 3, first_layer=True)
        assert "first_layer=True" in layer.extra_repr()
        assert "num_parallel=3" in layer.extra_repr()

    def test_gradients_flow_to_all_params(self):
        layer = BatchEnsembleLinear(8, 4, 3, first_layer=True)
        x = torch.randn(5, 8)
        loss = layer(x).pow(2).sum()
        loss.backward()
        for name in ("weight", "R", "S", "bias"):
            grad = getattr(layer, name).grad
            assert grad is not None, f"No grad on {name}"
            assert torch.any(grad != 0), f"Zero grad on {name}"

    def test_bias_false_omits_parameter(self):
        layer = BatchEnsembleLinear(8, 4, 3, bias=False)
        assert layer.bias is None
        param_names = {n for n, _ in layer.named_parameters()}
        assert "bias" not in param_names

    def test_2d_and_3d_inputs_agree(self):
        torch.manual_seed(0)
        layer = BatchEnsembleLinear(8, 4, 3, first_layer=True)
        layer.eval()
        x2d = torch.randn(6, 8)
        x3d = x2d.unsqueeze(0).expand(3, -1, -1).contiguous()
        out2d = layer(x2d)
        out3d = layer(x3d)
        assert torch.allclose(out2d, out3d, atol=1e-6)


# ===================================================================
# SNN predictor
# ===================================================================


class TestSNN:
    @pytest.mark.parametrize("bad_k", [0, 1, -3])
    def test_rejects_num_parallel_le_1(self, bad_k):
        with pytest.raises(ValueError, match="num_parallel >= 2"):
            SNN(
                input_dim=8,
                hidden_dims=[16, 8],
                num_endpoints=3,
                dropout=0.1,
                num_parallel=bad_k,
            )

    @pytest.mark.parametrize(
        "hidden_dims",
        [None, [], [32], [32, 16], [64, 64, 32]],
    )
    def test_forward_shape(self, hidden_dims):
        model = SNN(
            input_dim=12,
            hidden_dims=hidden_dims,
            num_endpoints=4,
            dropout=0.1,
            num_parallel=3,
        )
        model.eval()
        x = torch.randn(5, 12)
        out = model(x)
        assert out.shape == (5, 4)

    @pytest.mark.parametrize(
        "hidden_dims, expected_dim",
        [
            (None, 12),
            ([], 12),
            ([32], 32),
            ([32, 16], 16),
            ([64, 64, 8], 8),
        ],
    )
    def test_latent_dim(self, hidden_dims, expected_dim):
        model = SNN(
            input_dim=12,
            hidden_dims=hidden_dims,
            num_endpoints=4,
            dropout=0.0,
            num_parallel=2,
        )
        assert model.latent_dim == expected_dim

    def test_encode_output_shape(self):
        model = SNN(
            input_dim=12,
            hidden_dims=[32, 16],
            num_endpoints=4,
            dropout=0.0,
            num_parallel=3,
        )
        model.eval()
        x = torch.randn(5, 12)
        latent = model.encode(x)
        assert latent.shape == (5, 16)

    def test_encode_no_hidden_returns_input_shape(self):
        model = SNN(
            input_dim=12,
            hidden_dims=None,
            num_endpoints=4,
            dropout=0.0,
            num_parallel=2,
        )
        model.eval()
        x = torch.randn(5, 12)
        latent = model.encode(x)
        # No hidden layers: encode returns the raw input.
        assert latent.shape == (5, 12)
        assert torch.equal(latent, x)

    def test_first_layer_flag_only_on_layer_zero(self):
        model = SNN(
            input_dim=8,
            hidden_dims=[16, 8],
            num_endpoints=3,
            dropout=0.0,
            num_parallel=2,
        )
        flags = [layer.first_layer for layer in model.layers]
        assert flags[0] is True
        assert all(f is False for f in flags[1:])

    def test_param_efficiency_vs_naive_ensemble(self):
        common = dict(
            input_dim=32,
            hidden_dims=[64, 64],
            num_endpoints=4,
            dropout=0.0,
        )
        model_k2 = SNN(**common, num_parallel=2)
        model_k8 = SNN(**common, num_parallel=8)
        n_k2 = sum(p.numel() for p in model_k2.parameters())
        n_k8 = sum(p.numel() for p in model_k8.parameters())
        # A naive parallel ensemble scales linearly with k, so k=8 would be
        # ~4x the k=2 model. BatchEnsemble shares the dominant W term, so
        # k=8 should stay well under 2x the k=2 model.
        assert n_k8 < 2 * n_k2

    def test_mc_dropout_regression(self):
        """Fresh SELU/AlphaDropout per position — required for MC-dropout counting."""
        torch.manual_seed(0)
        model = SNN(
            input_dim=8,
            hidden_dims=[16, 16, 8],
            num_endpoints=3,
            dropout=0.5,
            num_parallel=3,
        )
        model.train()
        x = torch.randn(4, 8)
        out1 = model(x)
        out2 = model(x)
        # With dropout>0 in train mode, two calls must differ.
        assert not torch.allclose(out1, out2)

        # Distinct AlphaDropout instances — one per hidden position (3 here).
        dropout_ids = {id(m) for m in model.modules() if isinstance(m, nn.AlphaDropout)}
        assert len(dropout_ids) == 3

        # Control: dropout=0 gives deterministic output even in train mode.
        model0 = SNN(
            input_dim=8,
            hidden_dims=[16, 16, 8],
            num_endpoints=3,
            dropout=0.0,
            num_parallel=3,
        )
        model0.train()
        out_a = model0(x)
        out_b = model0(x)
        assert torch.allclose(out_a, out_b)

    def test_deterministic_eval(self):
        torch.manual_seed(0)
        model = SNN(
            input_dim=8,
            hidden_dims=[16, 8],
            num_endpoints=3,
            dropout=0.5,
            num_parallel=3,
        )
        model.eval()
        x = torch.randn(4, 8)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_backward_updates_shared_and_ensemble_params(self):
        model = SNN(
            input_dim=8,
            hidden_dims=[16, 8],
            num_endpoints=3,
            dropout=0.0,
            num_parallel=3,
        )
        x = torch.randn(4, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        first = model.layers[0]
        for name in ("weight", "R", "S", "bias"):
            grad = getattr(first, name).grad
            assert grad is not None, f"No grad on layers[0].{name}"
            assert torch.any(grad != 0), f"Zero grad on layers[0].{name}"

    def test_snn_has_no_reset_parameters_method(self):
        assert not hasattr(SNN, "_reset_parameters")
