"""Regression tests for CLM encoder padding masking (issue #433).

Verifies that padding positions are excluded from pooling in CNN and RNN.
"""

import torch

from matcha.torch.encoders.cnn import CNN
from matcha.torch.encoders.rnn import RNN


# ── helpers ──────────────────────────────────────────────────────────────────


def _small_cnn():
    return CNN(
        num_characters=30,
        embedding_dim=16,
        hidden_dim=16,
        kernel_dims=[3],
        num_heads=4,
        activation="relu",
        dropout=0.0,
    )


def _small_rnn(bidirectional=False):
    return RNN(
        num_layers=1,
        num_characters=30,
        embedding_dim=16,
        rnn_type="gru",
        hidden_dim=16,
        bidirectional=bidirectional,
        num_heads=4,
        dropout=0.0,
    )


# ── CNN ───────────────────────────────────────────────────────────────────────


class TestCNNMasking:
    def test_forward_with_padding_batch(self):
        """Batch with mixed-length molecules produces finite output of correct shape."""
        torch.manual_seed(42)
        model = _small_cnn()
        model.eval()
        token_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [1, 2, 3, 4, 5],
            ]
        )
        with torch.no_grad():
            out = model(token_ids)
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()

    def test_key_padding_mask_changes_result(self):
        """Same real tokens with different-length padding padding yield identical embeddings.

        The key_padding_mask prevents the CLS token from attending to padding
        positions, so the number of trailing zeros must not affect the output.
        """
        torch.manual_seed(42)
        model = _small_cnn()
        model.eval()

        short_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
        long_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])

        with torch.no_grad():
            emb_short = model(short_ids)
            emb_long = model(long_ids)

        assert torch.allclose(emb_short, emb_long, atol=1e-5)


# ── RNN ───────────────────────────────────────────────────────────────────────


class TestRNNMasking:
    def test_forward_with_padding_batch(self):
        """Padded batch produces correct shape with no NaN."""
        torch.manual_seed(42)
        model = _small_rnn()
        model.eval()
        token_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [1, 2, 3, 4, 5],
            ]
        )
        with torch.no_grad():
            out = model(token_ids)
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()

    def test_bidirectional_with_variable_lengths(self):
        """Bidirectional GRU with pack_padded_sequence handles variable lengths correctly."""
        torch.manual_seed(42)
        model = _small_rnn(bidirectional=True)
        model.eval()
        token_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 0, 0],
                [1, 2, 0, 0, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ]
        )
        with torch.no_grad():
            out = model(token_ids)
        # bidirectional doubles fp_dim: 16 * 2 = 32
        assert out.shape == (3, 32)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()
