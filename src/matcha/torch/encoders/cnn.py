"""1D Convolutional Neural Network (CNN) encoder for chemical language representations."""

import torch
import torch.nn as nn
from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.nn.activations import ActivationRegistry
from matcha.torch.encoders.base_encoder import BaseEncoder, EncoderRegistry


class Conv1d(nn.Module):
    """Single 1D convolution layer with dropout and activation.

    :param int input_dim: Number of input channels.
    :param int channel_dim: Number of output channels.
    :param int kernel_dim: Kernel size for the convolution.
    :param str activation: Activation function name.
    :param float dropout: Dropout rate applied after convolution.
    """

    def __init__(self, input_dim, channel_dim, kernel_dim, activation, dropout):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(input_dim, channel_dim, kernel_dim, padding="same")
        self.dropout = nn.Dropout(dropout)
        self.activation = ActivationRegistry[activation]()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return self.activation(x)


@EncoderRegistry.register()
class CNN(BaseEncoder, HyperparametersMixin):
    """1D convolution neural network (CNN) encoder for modelling chemical language
    representations. After processing the input sequence, self attention is used
    to compute a global representation of the molecule by making the [cls] token
    attend to each other position in the string.
    It inherits from  :class:`lightning.pytorch.core.mixins` for saving its hyperparameters,
    and from :class:`BaseEncoder` to be consistent with other encoders.
    References:
    - https://arxiv.org/abs/2407.12152
    - https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g

    It is intended to be used inside a :class:`BaseClassicModel` instance.
    Check the docs of :class:`matcha.torch.models.classic.CNNModel` for further details.

    :param int num_characters: total number of unique tokens in the dataset's
        dictionary

    :param int embedding_dim: number of token embedding features

    :param list[int] hidden_dims: number of hidden token dimensionality across each
        convolutional layer

    :param list[int] kernel_dims: kernel sizes for each convolutional layer

    :param int num_heads: number of self attention heads, must divide hidden_dims
        evenly

    :param str activation: activation function to use throughout all layers

    :param float dropout:  dropout noise level
    """

    def __init__(
        self,
        num_characters: int,
        embedding_dim: int,
        hidden_dim: int,
        kernel_dims: list[int],
        num_heads: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        # Snap num_heads to the largest divisor of hidden_dim that is <= num_heads,
        # so HPO can freely sample num_heads without causing a shape mismatch.
        while num_heads > 1 and hidden_dim % num_heads != 0:
            num_heads -= 1
        self.save_hyperparameters()
        self.embedding = nn.Embedding(num_characters, embedding_dim, padding_idx=0)

        conv_dims = [embedding_dim] + [hidden_dim] * len(kernel_dims)
        self.layers = torch.nn.ModuleList()
        for i in range(len(kernel_dims)):
            self.layers.append(
                Conv1d(
                    input_dim=conv_dims[i],
                    channel_dim=conv_dims[i + 1],
                    kernel_dim=kernel_dims[i],
                    activation=activation,
                    dropout=dropout,
                ),
            )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.layer_weights = nn.Parameter(torch.ones(len(kernel_dims)))
        self._fp_dim = hidden_dim

    def forward(self, token_ids: list[list[int]]) -> torch.Tensor:
        """Converts a nested list of integers objects into a (x, self.fp_dim) tensor for further
        processing.

        :param list[list[int]] token_ids: batched input from the dataloader

        :return torch.Tensor: learned representation
        """
        x = self.embedding(token_ids)
        x = self.pre_norm(x)
        x = torch.transpose(x, 1, 2)

        intermediate_reps = []

        for layer in self.layers:
            x = x + layer(x)
            intermediate_reps.append(x)

        weights = torch.softmax(self.layer_weights, dim=0)
        x = sum(w * rep for w, rep in zip(weights, intermediate_reps))

        x = torch.transpose(x, 1, 2)
        x = self.post_norm(x)
        key_padding_mask = token_ids == 0
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        return x[:, 0, :]
