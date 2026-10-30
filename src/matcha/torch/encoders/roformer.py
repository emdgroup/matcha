"""RoFormer transformer encoder with rotary position embeddings for chemical language."""

import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from transformers import AutoModel, RoFormerConfig
from matcha.torch.encoders.base_encoder import BaseEncoder, EncoderRegistry


@EncoderRegistry.register()
class RoFormer(BaseEncoder, HyperparametersMixin):
    """Transformer encoder (RoFormer) for modelling chemical language representations.
    It inherits from  :class:`lightning.pytorch.core.mixins` for saving its hyperparameters,
    and from :class:`BaseEncoder` to be consistent with other encoders.
    References:
    - https://arxiv.org/abs/1810.04805
    - https://arxiv.org/abs/2106.09553

    It is intended to be used inside a :class:`BaseClassicModel` instance.
    Check the docs of :class:`matcha.torch.models.classic.TransformerModel` for further details.

    :param int num_characters: dictionary size

    :param int hidden_dim: token embedding dimensionality, defaults to
        768

    :param int expansion_dim: dimensionality to use inside transformer blocks,
        defaults to 3072

    :param int num_heads: number of attention heads for computing self-attention,
        defaults to 12

    :param int num_layers: number of transformer blocks to use, defaults
        to 4

    :param float attention_dropout: dropout to use when computing attention,
        defaults to 0.1

    :param float hidden_dropout: dropout to use in dense layers inside
        transformer blocks, defaults to 0.1
    """

    def __init__(
        self,
        num_characters: int,
        hidden_dim: int = 768,
        expansion_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 4,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_config = RoFormerConfig(
            vocab_size=num_characters,
            hidden_size=hidden_dim,
            intermediate_size=expansion_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=attention_dropout,
            pad_token_id=0,
            cls_token_id=2,
            rotary_value=True,
        )

        self.model = AutoModel.from_config(model_config)
        self.model_config = model_config
        self._fp_dim = self.model.config.hidden_size
        self.layers = self.model.encoder.layer

    def forward_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized SMILES into per-token contextual embeddings.

        Consumed by both the classic path (which takes the [CLS] slice via
        :meth:`forward`) and the MLM pretraining path (which keeps the full
        sequence output).

        :param torch.Tensor token_ids: Batched token IDs [batch_size, seq_len].
        :returns: Per-token embeddings [batch_size, seq_len, hidden_dim].
        :rtype: torch.Tensor
        """
        attention_mask = (token_ids != 0).long()
        out = self.model(token_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized SMILES into a fixed-size molecular representation.

        Uses the [CLS] token output from the last hidden layer.

        :param torch.Tensor token_ids: Batched token IDs [batch_size, seq_len].
        :returns: Learned molecular representation [batch_size, hidden_dim].
        :rtype: torch.Tensor
        """
        return self.forward_tokens(token_ids)[:, 0, :]
