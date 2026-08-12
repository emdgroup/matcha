"""Unit tests for the SNN predictor and its BatchEnsembleLinear primitive."""

import math

import pytest
import torch

from matcha.torch.predictors.snn import BatchEnsembleLinear


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
