"""Tests for :class:`matcha.torch.models.finetuning.PretrainedEncoderWrapper`.

Regression coverage for the MLM path: the wrapper must return a flat
``[batch_size, hidden_dim]`` embedding when adapting an MLM encoder. An
earlier bug called ``encoder(token_ids)`` (which already CLS-pools for the
classic path) and then sliced ``[:, 0, :]`` on the pooled 2D tensor, raising
``IndexError: too many indices for tensor of dimension 2`` on the first
finetuning step. The fix routes through ``encoder.forward_tokens`` so the
wrapper is the sole owner of the CLS slice.
"""

import torch

from matcha.torch.encoders.roformer import RoFormer
from matcha.torch.models.finetuning.pretrained_encoder_wrapper import (
    PretrainedEncoderWrapper,
)


_VOCAB_SIZE = 32
_HIDDEN_DIM = 16
_EXPANSION_DIM = 32
_NUM_HEADS = 2
_NUM_LAYERS = 2


def _make_roformer() -> RoFormer:
    return RoFormer(
        num_characters=_VOCAB_SIZE,
        hidden_dim=_HIDDEN_DIM,
        expansion_dim=_EXPANSION_DIM,
        num_heads=_NUM_HEADS,
        num_layers=_NUM_LAYERS,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


def test_mlm_encode_returns_flat_embedding():
    """MLM-typed wrapper must CLS-pool to ``[batch_size, hidden_dim]``."""
    torch.manual_seed(0)
    encoder = _make_roformer()
    encoder.eval()
    wrapper = PretrainedEncoderWrapper(encoder, encoder_type="mlm")

    batch_size, seq_len = 3, 7
    tokens = torch.randint(low=1, high=_VOCAB_SIZE, size=(batch_size, seq_len))

    with torch.no_grad():
        embedding = wrapper.encode({"token_ids": tokens})

    assert embedding.shape == (batch_size, _HIDDEN_DIM)


def test_mlm_encode_matches_forward_tokens_cls_slice():
    """The wrapper's MLM output must equal ``forward_tokens(x)[:, 0, :]``.

    Structural check that the wrapper owns the CLS slice and does not
    double-pool by calling the encoder's classic ``forward``.
    """
    torch.manual_seed(0)
    encoder = _make_roformer()
    encoder.eval()
    wrapper = PretrainedEncoderWrapper(encoder, encoder_type="mlm")

    tokens = torch.randint(low=1, high=_VOCAB_SIZE, size=(2, 5))
    with torch.no_grad():
        expected = encoder.forward_tokens(tokens)[:, 0, :]
        actual = wrapper.encode({"token_ids": tokens})

    assert torch.allclose(actual, expected, atol=1e-6)
