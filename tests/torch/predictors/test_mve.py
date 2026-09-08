"""Unit tests for the MVEPredictor head."""

import pytest
import torch

from matcha.torch.predictors.base_predictor import PredictorRegistry
from matcha.torch.predictors.mve import MVEPredictor


class TestMVEPredictorRegistry:
    def test_registered_under_mve_alias(self):
        assert "mve" in PredictorRegistry
        assert PredictorRegistry["mve"] is MVEPredictor

    def test_returns_mean_and_log_var_marker(self):
        assert MVEPredictor._returns_mean_and_log_var is True


class TestMVEPredictor:
    @pytest.mark.parametrize(
        "hidden_dims",
        [None, [], [32], [32, 16], [64, 32, 16]],
    )
    @pytest.mark.parametrize("num_endpoints", [1, 3, 7])
    def test_forward_shape_is_double_num_endpoints(self, hidden_dims, num_endpoints):
        input_dim = 12
        batch = 5
        head = MVEPredictor(
            input_dim=input_dim,
            hidden_dims=hidden_dims if hidden_dims else None,
            num_endpoints=num_endpoints,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        head.eval()
        out = head(torch.randn(batch, input_dim))
        assert out.shape == (batch, 2 * num_endpoints)

    @pytest.mark.parametrize(
        "hidden_dims, expected_latent",
        [
            (None, 12),
            ([32], 32),
            ([32, 16], 16),
            ([64, 8, 4], 4),
        ],
    )
    def test_latent_dim(self, hidden_dims, expected_latent):
        head = MVEPredictor(
            input_dim=12,
            hidden_dims=hidden_dims,
            num_endpoints=3,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        assert head.latent_dim == expected_latent

    def test_encode_returns_body_output(self):
        head = MVEPredictor(
            input_dim=12,
            hidden_dims=[32, 16],
            num_endpoints=3,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        head.eval()
        x = torch.randn(5, 12)
        latent = head.encode(x)
        assert latent.shape == (5, 16)

    def test_encode_without_hidden_returns_input(self):
        head = MVEPredictor(
            input_dim=12,
            hidden_dims=None,
            num_endpoints=3,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        head.eval()
        x = torch.randn(5, 12)
        latent = head.encode(x)
        assert torch.equal(latent, x)

    def test_gradients_flow_to_both_heads(self):
        head = MVEPredictor(
            input_dim=8,
            hidden_dims=[16],
            num_endpoints=2,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        x = torch.randn(4, 8)
        out = head(x)
        loss = out.pow(2).sum()
        loss.backward()
        for name in ("mean_head", "log_var_head"):
            module = getattr(head, name)
            for p in module.parameters():
                assert p.grad is not None
                assert torch.any(p.grad != 0), f"No grad on {name}"

    def test_mean_and_log_var_are_distinct_linear_layers(self):
        head = MVEPredictor(
            input_dim=8,
            hidden_dims=None,
            num_endpoints=3,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        assert head.mean_head is not head.log_var_head
        assert not torch.equal(head.mean_head.weight, head.log_var_head.weight)

    def test_output_splits_into_mean_and_log_var_halves(self):
        """The forward output must be a concatenation of mean and log_var,
        so slicing to num_endpoints must equal the mean head's output."""
        head = MVEPredictor(
            input_dim=8,
            hidden_dims=None,
            num_endpoints=3,
            dropout=0.0,
            activation="relu",
            norm=None,
        )
        head.eval()
        x = torch.randn(4, 8)
        out = head(x)
        expected_mean = head.mean_head(x)
        expected_log_var = head.log_var_head(x)
        assert torch.allclose(out[:, :3], expected_mean, atol=1e-6)
        assert torch.allclose(out[:, 3:], expected_log_var, atol=1e-6)
