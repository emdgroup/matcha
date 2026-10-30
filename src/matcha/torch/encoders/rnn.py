"""Recurrent Neural Network (RNN) encoder for chemical language representations."""

import torch
import torch.nn as nn
from lightning.pytorch.core.mixins import HyperparametersMixin
from matcha.torch.encoders.base_encoder import BaseEncoder, EncoderRegistry


@EncoderRegistry.register()
class RNN(BaseEncoder, HyperparametersMixin):
    """Recurrent Neural Network (RNN) encoder for modelling chemical language
    representations. After processing the input sequence, self attention is used
    to compute a global representation of the molecule by making the [cls] token
    attend to each other position in the string.
    It inherits from  :class:`lightning.pytorch.core.mixins` to save its hyperparameters,
    and from :class:`BaseEncoder` to be consistent with other encoders.
    It is intended to be used inside a :class:`matcha.torch.models.classic.base_classic_model.BaseClassicModel`
    instance.
    References:
    - https://arxiv.org/abs/2407.12152
    - https://www.sciencedirect.com/science/article/pii/S2667318521000143

    It is intended to be used inside a :class:`BaseClassicModel` instance.
    Check the docs of :class:`matcha.torch.models.classic.RNNModel` for further details.

    :param int num_layers: number of RNN layers

    :param int num_characters: total number of unique tokens in the dataset's
        dictionary

    :param int embedding_dim: number of token embedding features

    :param str rnn_type: whether to use LSTM or GRU architectures

    :param int hidden_dim: number of hidden token dimensionality

    :param str bidirectional: whether to concatenate representation obtained
        while running RNN in reverse on the string

    :param int num_heads: number of self attention heads, must divide hidden_dims
        evenly

    :param float dropout:  dropout noise level
    """

    def __init__(
        self,
        num_layers: int,
        num_characters: int,
        embedding_dim: int,
        rnn_type: str,
        hidden_dim: int,
        bidirectional: str,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_characters, embedding_dim, padding_idx=0)

        if rnn_type.lower() == "gru":
            rnn = nn.GRU
        elif rnn_type.lower() == "lstm":
            rnn = nn.LSTM

        self.rnn = rnn(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.layers = nn.ModuleList([self.rnn])
        if bidirectional:
            self._fp_dim = hidden_dim * 2
        else:
            self._fp_dim = hidden_dim

        # Snap num_heads to the largest divisor of fp_dim that is <= num_heads,
        # so HPO can freely sample both hidden_dim and num_heads independently.
        while num_heads > 1 and self._fp_dim % num_heads != 0:
            num_heads -= 1
        self.save_hyperparameters()

        self.attention = nn.MultiheadAttention(
            embed_dim=self.fp_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_in = nn.LayerNorm(embedding_dim)
        self.norm_out = nn.LayerNorm(self._fp_dim)

    def forward(self, token_ids: list[list[int]]) -> torch.Tensor:
        """Converts a nested list of integers objects into a (x, self.fp_dim) tensor for further
        processing.

        :param list[list[int]] token_ids: batched input from the dataloader

        :return torch.Tensor: learned representation
        """
        x = self.embedding(token_ids)
        x = self.norm_in(x)
        lengths = (token_ids != 0).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed, _ = self.rnn(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=token_ids.size(1)
        )
        x = self.norm_out(x)
        key_padding_mask = token_ids == 0
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        return x[:, 0, :]
