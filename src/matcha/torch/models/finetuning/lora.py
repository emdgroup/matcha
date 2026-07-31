"""Low-Rank Adaptation (LoRA) for parameter-efficient finetuning.

Implements the LoRA technique (Hu et al., arXiv:2106.09685) which decomposes
weight updates into low-rank matrices, enabling efficient adaptation of large
pretrained models while keeping most parameters frozen.
"""

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper around a frozen ``nn.Linear`` layer.

    Decomposes the weight update into two low-rank matrices so that the original
    pretrained weights are kept frozen while only the small adapter matrices are
    trained.  The effective weight becomes ``W' = W + (B @ A) * scaling``.

    References:
    - https://arxiv.org/abs/2106.09685

    :param nn.Linear original_linear: the frozen linear layer to wrap
    :param int rank: rank of the low-rank decomposition
    :param float alpha: scaling numerator (``scaling = alpha / rank``)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # Low-rank decomposition: W' = W + (B @ A) * scaling
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the original linear output plus the low-rank adaptation.

        :param torch.Tensor x: input tensor
        :returns: ``W(x) + (B @ A)(x) * scaling``
        :rtype: torch.Tensor
        """
        base_out = self.original_linear(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    min_dim: int = 64,
) -> list[nn.Parameter]:
    """Recursively replace ``nn.Linear`` layers inside *module* with
    :class:`LoRALinear` wrappers.

    Only targets linear layers whose *both* dimensions are ``>= min_dim`` so
    that tiny projection heads, edge-feature layers, etc. are left untouched.

    :param nn.Module module: the module to modify **in-place**
    :param int rank: LoRA rank
    :param float alpha: LoRA scaling numerator
    :param int min_dim: minimum feature dimension for a layer to be wrapped

    :return list[nn.Parameter]: all newly-created LoRA parameters
    """
    lora_params: list[nn.Parameter] = []
    for name, child in list(module.named_children()):
        if (
            isinstance(child, nn.Linear)
            and child.in_features >= min_dim
            and child.out_features >= min_dim
        ):
            lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
            setattr(module, name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        else:
            lora_params.extend(apply_lora(child, rank, alpha, min_dim))
    return lora_params
