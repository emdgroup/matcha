"""Tests for :class:`matcha.torch.models.pretraining.roformer_mlm.RoFormerMLM`.

Focus of these tests is the Stage-4 unification from issue #24: after deleting
``RoFormerMLMEncoder``, :class:`RoFormerMLM` must:

- instantiate the canonical :class:`RoFormer` encoder;
- consume per-token embeddings through ``encoder.forward_tokens``;
- still produce MLM logits of shape ``[batch_size, seq_len, vocab_size]``.
"""

import torch

from matcha.torch.encoders.roformer import RoFormer
from matcha.torch.models.pretraining.roformer_mlm import RoFormerMLM


_VOCAB_SIZE = 32
_HIDDEN_DIM = 16
_EXPANSION_DIM = 32
_NUM_HEADS = 2
_NUM_LAYERS = 2


def _make_model(**overrides) -> RoFormerMLM:
    """Build a tiny RoFormerMLM for testing."""
    kwargs = dict(
        enc_num_characters=_VOCAB_SIZE,
        enc_hidden_dim=_HIDDEN_DIM,
        enc_expansion_dim=_EXPANSION_DIM,
        enc_num_heads=_NUM_HEADS,
        enc_num_layers=_NUM_LAYERS,
        enc_attention_dropout=0.0,
        enc_hidden_dropout=0.0,
        pred_hidden_dims=[8],
        pred_activation="gelu",
        pred_dropout=0.0,
    )
    kwargs.update(overrides)
    return RoFormerMLM(**kwargs)


def _make_batch(batch_size: int = 2, seq_len: int = 5) -> dict:
    tokens = torch.randint(low=1, high=_VOCAB_SIZE, size=(batch_size, seq_len))
    y = torch.randint(low=0, high=_VOCAB_SIZE, size=(batch_size, seq_len))
    return {"token_ids": tokens, "y": y}


def test_encoder_is_canonical_roformer():
    """The pretraining model wires up the canonical ``RoFormer`` encoder."""
    model = _make_model()
    assert isinstance(model.encoder, RoFormer)


def test_forward_returns_per_token_logits():
    """End-to-end forward returns ``[batch_size, seq_len, vocab_size]`` logits."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()

    batch_size, seq_len = 2, 5
    batch = _make_batch(batch_size=batch_size, seq_len=seq_len)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (batch_size, seq_len, _VOCAB_SIZE)


def test_encode_returns_per_token_embeddings():
    """``encode`` returns ``[batch_size, seq_len, hidden_dim]`` from ``forward_tokens``."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()

    batch_size, seq_len = 2, 5
    batch = _make_batch(batch_size=batch_size, seq_len=seq_len)
    with torch.no_grad():
        out = model.encode(batch)
        expected = model.encoder.forward_tokens(batch["token_ids"])

    assert out.shape == (batch_size, seq_len, _HIDDEN_DIM)
    assert torch.allclose(out, expected, atol=1e-6)
