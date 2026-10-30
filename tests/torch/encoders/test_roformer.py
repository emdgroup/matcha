"""Tests for :class:`matcha.torch.encoders.roformer.RoFormer`.

Covers the Stage-4 unification from issue #24: the canonical RoFormer exposes
``forward_tokens`` (per-token embeddings) so both the classic path (which
returns the [CLS] slice via ``forward``) and the MLM pretraining path share
the same encoder implementation.
"""

import torch

from matcha.torch.encoders.roformer import RoFormer


_VOCAB_SIZE = 32
_HIDDEN_DIM = 16
_EXPANSION_DIM = 32
_NUM_HEADS = 2
_NUM_LAYERS = 2


def _make_roformer(**overrides) -> RoFormer:
    """Build a tiny RoFormer encoder for testing."""
    kwargs = dict(
        num_characters=_VOCAB_SIZE,
        hidden_dim=_HIDDEN_DIM,
        expansion_dim=_EXPANSION_DIM,
        num_heads=_NUM_HEADS,
        num_layers=_NUM_LAYERS,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    kwargs.update(overrides)
    return RoFormer(**kwargs)


def _make_tokens(batch_size: int = 2, seq_len: int = 5) -> torch.Tensor:
    """Random token IDs with all-valid (non-pad) positions."""
    return torch.randint(low=1, high=_VOCAB_SIZE, size=(batch_size, seq_len))


def test_forward_tokens_returns_per_token_embeddings():
    """``forward_tokens`` returns ``[batch_size, seq_len, hidden_dim]``."""
    torch.manual_seed(0)
    encoder = _make_roformer()
    encoder.eval()

    batch_size, seq_len = 2, 5
    tokens = _make_tokens(batch_size=batch_size, seq_len=seq_len)
    with torch.no_grad():
        out = encoder.forward_tokens(tokens)

    assert out.shape == (batch_size, seq_len, _HIDDEN_DIM)


def test_forward_equals_forward_tokens_cls_slice():
    """``forward(x)`` must equal ``forward_tokens(x)[:, 0, :]``.

    This is the structural check that the classic path is a thin view over
    ``forward_tokens`` — the same code path used by the MLM pretraining head.
    """
    torch.manual_seed(0)
    encoder = _make_roformer()
    encoder.eval()

    tokens = _make_tokens()
    with torch.no_grad():
        per_token = encoder.forward_tokens(tokens)
        cls_slice = per_token[:, 0, :]
        cls_from_forward = encoder(tokens)

    assert torch.allclose(cls_from_forward, cls_slice, atol=1e-6)
    assert cls_from_forward.shape == (tokens.size(0), _HIDDEN_DIM)


def test_fp_dim_matches_hidden_dim():
    """``fp_dim`` still reports the hidden dimensionality after the refactor."""
    encoder = _make_roformer()
    assert encoder.fp_dim == _HIDDEN_DIM
