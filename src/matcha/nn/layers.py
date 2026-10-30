"""Layer primitives for tabular and graph neural networks."""

import math

import torch
from torch import nn
from torch_geometric.nn import norm
from matcha.nn.activations import ActivationRegistry
from matcha.utils.registry import ClassRegistry

LayerRegistry = ClassRegistry()

# ------------------------------------------------------------------------------#
#   Tabular layers
# ------------------------------------------------------------------------------#


@LayerRegistry.register("adarmsn")
class AdaRMSN(nn.Module):
    """
    Adaptive Root Mean Square Normalization (AdaRMSN)

    This normalization preserves token magnitude information while controlling
    token magnitudes to prevent softmax saturation.
    """

    def __init__(self, dim):
        """
        :param int dim: Feature dimension to normalize over.
        """
        super(AdaRMSN, self).__init__()
        self.dim = dim
        self.beta = nn.Parameter(torch.ones(dim))
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self, x) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor of shape ``(..., dim)``.
        :returns: Normalized tensor of the same shape.
        :rtype: torch.Tensor
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        x_normalized = x / (norm + 1e-8)
        alpha_x_plus_beta_norm = torch.norm(
            self.alpha * x + self.beta, dim=-1, keepdim=True
        )
        gamma_prime = alpha_x_plus_beta_norm / math.sqrt(self.dim)
        return x_normalized * gamma_prime / math.sqrt(self.dim)


@LayerRegistry.register("batch")
class BatchNorm(nn.BatchNorm1d):
    """Batch normalization (wraps :class:`torch.nn.BatchNorm1d`)."""

    pass


@LayerRegistry.register("layer")
class LayerNorm(nn.LayerNorm):
    """Layer normalization (wraps :class:`torch.nn.LayerNorm`)."""

    pass


@LayerRegistry.register("instance")
class InstanceNorm(nn.InstanceNorm1d):
    """Instance normalization (wraps :class:`torch.nn.InstanceNorm1d`)."""

    pass


@LayerRegistry.register("graph")
class GraphNorm(norm.GraphNorm):
    """Graph normalization (wraps :class:`torch_geometric.nn.norm.GraphNorm`)."""

    pass


@LayerRegistry.register("lnbndr")
class LnBnDr(nn.Module):
    """Linear-Norm-Dropout-Activation layer stack to simplify using this
    pattern across different architectures.
    Reference: https://docs.fast.ai/layers.html#linbndrop
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        activation: str | None,
        norm: str,
    ):
        """
        :param int input_dim: Input feature dimension.
        :param int output_dim: Output feature dimension.
        :param float dropout: Dropout probability.
        :param activation: Activation name from :data:`ActivationRegistry`, or ``None``.
        :type activation: str or None
        :param str norm: Normalization name from :data:`LayerRegistry`.
        """
        super(LnBnDr, self).__init__()
        layers = [nn.Linear(input_dim, output_dim)]

        if norm is not None:
            layers.append(LayerRegistry[norm](output_dim))

        layers.append(nn.Dropout(dropout))

        if activation is not None:
            layers.append(ActivationRegistry[activation]())

        self.layers = nn.Sequential(*layers)
        self.in_features = input_dim
        self.out_features = output_dim

    def forward(self, x) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor of shape ``(batch, input_dim)``.
        :returns: Output tensor of shape ``(batch, output_dim)``.
        :rtype: torch.Tensor
        """
        return self.layers(x)


@LayerRegistry.register("multiln")
class MultiLn(nn.Module):
    """Parallel linear layer implementation, used in the task-specific MLPs
    when training multitask models.
    Adapted from: https://github.com/datamol-io/graphium/blob/main/graphium/nn/ensemble_layers.py
    """

    def __init__(self, in_dim: int, out_dim: int, num_parallel: int, bias: bool = True):
        """
        :param int in_dim: Input feature dimension.
        :param int out_dim: Output feature dimension.
        :param int num_parallel: Number of parallel linear heads.
        :param bool bias: Whether to include a bias term.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parallel, out_dim, in_dim))
        self.bias = (
            nn.Parameter(torch.Tensor(num_parallel, 1, out_dim)) if bias else None
        )
        self.init_fn = nn.init.xavier_uniform_
        self.reset_parameters()

    def reset_parameters(self):
        self.init_fn(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor of shape ``(num_parallel, batch, in_dim)``.
        :returns: Output tensor of shape ``(num_parallel, batch, out_dim)``.
        :rtype: torch.Tensor
        """
        out = torch.matmul(self.weight, x.transpose(-1, -2)).transpose(-1, -2)
        if self.bias is not None:
            out += self.bias
        return out


@LayerRegistry.register("multibatch")
class MultiBatchNorm(nn.Module):
    """Analogue of BatchNorm1d for working with MultiLn layers,
    keeping separate statistics for each head."""

    def __init__(
        self,
        num_features: int,
        num_parallel: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        """
        :param int num_features: Number of features per head.
        :param int num_parallel: Number of parallel heads.
        :param float eps: Value added to denominator for numerical stability.
        :param float momentum: Value for running mean/variance computation.
        """
        super().__init__()
        self.num_parallel = num_parallel
        self.num_features = num_features

        self.weight = nn.Parameter(torch.ones(num_parallel, num_features))
        self.bias = nn.Parameter(torch.zeros(num_parallel, num_features))

        self.register_buffer("running_mean", torch.zeros(num_parallel, num_features))
        self.register_buffer("running_var", torch.ones(num_parallel, num_features))

        self.eps = eps
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor of shape ``(num_parallel, batch, num_features)``.
        :returns: Normalized tensor of the same shape.
        :rtype: torch.Tensor
        """
        if self.training:
            batch_mean = x.mean(dim=1)
            batch_var = x.var(dim=1, unbiased=False)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var.detach()
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x_hat = (x - batch_mean[:, None, :]) / torch.sqrt(
            batch_var[:, None, :] + self.eps
        )
        out = self.weight[:, None, :] * x_hat + self.bias[:, None, :]
        return out


@LayerRegistry.register("multimlp")
class MultiMLP(nn.Module):
    """Stack of MultiLn, norm (if batch, then its multibatch layers), dropout
    and activation layers for instantiating the MLPs for each head in multitask
    networks"""

    def __init__(
        self,
        input_dim,
        dims: list[int],
        num_parallel: int,
        dropout: float,
        activation: str,
        norm: str,
    ):
        """
        :param int input_dim: Input feature dimension.
        :param list[int] dims: Hidden layer dimensions.
        :param int num_parallel: Number of parallel MLP heads.
        :param float dropout: Dropout probability.
        :param str activation: Activation name from :data:`ActivationRegistry`.
        :param str norm: Normalization name from :data:`LayerRegistry`.
        """
        super().__init__()
        self.num_layers = len(dims)
        self.linear_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])
        dims = [input_dim] + dims

        for i in range(len(dims) - 1):
            self.linear_layers.append(MultiLn(dims[i], dims[i + 1], num_parallel))
            self.norm_layers.append(LayerRegistry[norm](dims[i + 1], num_parallel))

        self.linear_layers.append(MultiLn(dims[-1], 1, num_parallel))

        self.act_fn = ActivationRegistry[activation]()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor of shape ``(num_parallel, batch, input_dim)``.
        :returns: Output tensor of shape ``(batch, num_parallel)``.
        :rtype: torch.Tensor
        """
        for i in range(self.num_layers):
            x = self.linear_layers[i](x)
            x = self.norm_layers[i](x)
            x = self.dropout(x)
            x = self.act_fn(x)
        x = self.linear_layers[-1](x)
        return x.transpose(1, 0).squeeze(-1)


# ------------------------------------------------------------------------------#
#   2D Graph layers
# ------------------------------------------------------------------------------#


def from_dense_batch(out, batch):
    """Reshape a dense-batched tensor back to sparse (variable-length) node format.

    Converts from shape ``(batch_size, max_num_atoms, feat_size)`` to
    ``(total_nodes, feat_size)`` using graph membership indices.

    :param torch.Tensor out: Dense tensor of shape ``(batch_size, max_num_nodes, feat_size)``.
    :param torch.Tensor batch: Node-to-graph assignment vector of shape ``(total_nodes,)``.
    :returns: Reconstructed sparse tensor of shape ``(total_nodes, feat_size)``.
    :rtype: torch.Tensor
    :raises ValueError: If any node index exceeds the maximum number of nodes in the batch.
    """
    batch_size, max_num_nodes, feat_size = out.size()
    device = out.device
    num_nodes = torch.bincount(batch, minlength=batch_size)
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1, device=device), num_nodes.cumsum(dim=0)]
    )
    node_idx = torch.arange(batch.size(0), device=device) - cum_num_nodes[batch]
    max_nodes_in_batch = out.size(1)
    if node_idx.max() >= max_nodes_in_batch:
        raise ValueError("Node index exceeds maximum number of nodes in the batch.")
    x_reconstructed = out[batch, node_idx]

    return x_reconstructed


@LayerRegistry.register("spatial_encoder")
class SpatialEncoder(nn.Module):
    """PyTorch Geometric compatible Spatial Encoder.

    Encodes shortest path distances between node pairs as learnable embeddings.
    This is a replacement for dgl.nn.pytorch.SpatialEncoder.

    :param int max_dist: Maximum distance to encode (distances beyond this are clamped)
    :param int num_heads: Number of attention heads (embedding dimension)
    """

    def __init__(self, max_dist: int, num_heads: int):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        # +2 for: 0 (self-loop), distances 1 to max_dist, and max_dist+1 (for distances > max_dist)
        self.embedding = nn.Embedding(max_dist + 2, num_heads)

    def forward(self, spd: torch.Tensor) -> torch.Tensor:
        """Encode shortest path distances.

        :param torch.Tensor spd: Shortest path distance matrix [batch_size, max_nodes, max_nodes]
        :return torch.Tensor: Distance bias for attention [batch_size, max_nodes, max_nodes, num_heads]
        """
        # Clamp handles [0, max_dist + 1]; the where routes -1 (unreachable)
        # to the max_dist + 1 sentinel bucket.
        clamped = torch.clamp(spd, min=0, max=self.max_dist + 1)
        clamped = torch.where(spd < 0, torch.full_like(spd, self.max_dist + 1), clamped)

        return self.embedding(clamped.long())


def _gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Gaussian basis kernel function.

    :param torch.Tensor x: Input tensor
    :param torch.Tensor mean: Mean of the Gaussian
    :param torch.Tensor std: Standard deviation of the Gaussian
    :return torch.Tensor: Gaussian kernel values
    """
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


@LayerRegistry.register("spatial_encoder_3d")
class SpatialEncoder3d(nn.Module):
    """PyTorch Geometric compatible 3D Spatial Encoder.

    This module encodes pair-wise relation between node pair (i,j) in
    the 3D geometric space, according to the Gaussian Basis Kernel function.

    This is a replacement for dgl.nn.pytorch.SpatialEncoder3d from the paper:
    "One Transformer Can Understand Both 2D & 3D Molecular Data"
    https://arxiv.org/pdf/2210.01765.pdf

    The encoding is computed as:

    ψ_{(i,j)}^k = (1 / (√(2π) |σ^k|)) * exp(-0.5 * ((γ_{(i,j)} * ||r_i - r_j|| + β_{(i,j)} - μ^k) / |σ^k|)^2)

    where K is the number of Gaussian Basis kernels, r_i is the Cartesian coordinate
    of node i, γ and β are learnable scaling factors and biases determined by node types,
    and μ^k, σ^k are learnable centers and standard deviations of the kernels.

    :param int num_kernels: Number of Gaussian Basis Kernels
    :param int num_heads: Number of attention heads (default: 1)
    :param int atom_feat_dim: Dimensionality of projected atom features
    :param float max_dist: Upper bound of the deterministic linspace init for kernel means (default: 10.0)
    """

    def __init__(
        self,
        num_kernels: int,
        num_heads: int = 1,
        *,
        atom_feat_dim: int,
        max_dist: float = 10.0,
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.num_heads = num_heads
        self.max_dist = max_dist

        # Learnable kernel parameters — softplus keeps stds strictly positive
        # without the +1e-2 hack in forward.
        self.means = nn.Parameter(torch.empty(num_kernels))
        self.raw_stds = nn.Parameter(torch.empty(num_kernels))

        # Linear projections for Gaussian kernels
        self.linear_layer_1 = nn.Linear(num_kernels, num_kernels)
        self.linear_layer_2 = nn.Linear(num_kernels, num_heads)

        # Per-atom linear projections for distance scaling (gamma) and shift (beta)
        self.gamma_proj = nn.Linear(atom_feat_dim, 1)
        self.beta_proj = nn.Linear(atom_feat_dim, 1)

        # Deterministic kernel-centre spacing over [0, max_dist]; softplus(0.5413) ≈ 1.0.
        self.means.data.copy_(torch.linspace(0.0, max_dist, num_kernels))
        nn.init.constant_(self.raw_stds, 0.5413)
        nn.init.xavier_uniform_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, coord: torch.Tensor, atom_feats: torch.Tensor) -> torch.Tensor:
        """Encode 3D spatial relationships.

        :param torch.Tensor coord: 3D coordinates of nodes [batch_size, max_nodes, 3]
        :param torch.Tensor atom_feats: Projected atom features [batch_size, max_nodes, atom_feat_dim]
        :return torch.Tensor: Attention bias from 3D spatial encoding [batch_size, max_nodes, max_nodes, num_heads]
        """
        # Compute pairwise Euclidean distances
        euc_dist = torch.cdist(coord, coord, p=2.0)  # [B, N, N]

        # Compute per-atom scaling scalars from atom features
        gamma = self.gamma_proj(atom_feats)  # [B, N, 1]
        beta = self.beta_proj(atom_feats)  # [B, N, 1]

        # Additive pairwise combination (src + tgt), matching original embedding-sum design
        gamma_ij = gamma.unsqueeze(2) + gamma.unsqueeze(1)  # [B, N, N, 1]
        beta_ij = beta.unsqueeze(2) + beta.unsqueeze(1)  # [B, N, N, 1]

        # Scale euclidean distance
        euc_dist = gamma_ij * euc_dist.unsqueeze(-1) + beta_ij  # [B, N, N, 1]

        # Expand for all kernels
        euc_dist = euc_dist.expand(-1, -1, -1, self.num_kernels)

        # Apply Gaussian basis kernel
        gaussian_kernel = _gaussian(
            euc_dist, self.means, torch.nn.functional.softplus(self.raw_stds)
        )  # shape: [B, N, N, K]

        # Linear projection
        encoding = self.linear_layer_1(gaussian_kernel)
        encoding = torch.nn.functional.gelu(encoding)
        encoding = self.linear_layer_2(encoding)  # shape: [B, N, N, H]

        return encoding


@LayerRegistry.register("biased_mha")
class BiasedMultiHeadAttention(nn.Module):
    """Multi-head attention with optional distance bias.

    PyTorch Geometric compatible implementation of biased multi-head attention
    for graph transformers. This is a replacement for dgl.nn.pytorch.gt.BiasedMHA.

    :param int embed_dim: Embedding dimension
    :param int num_heads: Number of attention heads
    :param float dropout: Attention dropout ratio
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        :param torch.Tensor x: Input tensor [batch_size, seq_len, embed_dim]
        :param torch.Tensor attn_bias: Optional attention bias [batch_size, seq_len, seq_len, num_heads]
        :param torch.Tensor attn_mask: Optional attention mask [batch_size, seq_len] (True for valid positions)
        :return torch.Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add distance bias if provided
        if attn_bias is not None:
            # attn_bias: [batch, seq, seq, heads] -> [batch, heads, seq, seq]
            attn_bias = attn_bias.permute(0, 3, 1, 2)
            attn_scores = attn_scores + attn_bias

        # Apply mask if provided (safe-softmax: also zero out fully-padded
        # query rows before softmax to avoid NaN gradients through WV / WO).
        query_mask: torch.Tensor | None = None
        if attn_mask is not None:
            # attn_mask: [batch, seq] (True for valid positions)
            key_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq]
            query_mask = attn_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, seq, 1]
            attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))
            # For rows where all keys are masked (fully-padded queries),
            # replace the entire row with a finite constant so softmax
            # returns a valid (uniform) distribution instead of NaN.
            attn_scores = attn_scores.masked_fill(~query_mask, 0.0)

        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        if query_mask is not None:
            attn_weights = attn_weights.masked_fill(~query_mask, 0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out
