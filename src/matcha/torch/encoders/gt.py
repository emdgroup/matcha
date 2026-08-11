"""Graph Transformer (GT) encoder with distance-biased dense attention."""

import math
from typing import Optional

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from matcha.nn.layers import LnBnDr, SpatialEncoder, from_dense_batch
from matcha.utils.logging import get_default_logger

_LOGGER = get_default_logger(__name__)


class GTConv(nn.Module):
    """Graph Transformer Convolution layer with dense attention.

    Uses dense multi-head attention over ALL node pairs with distance bias,
    plus sparse edge bias that only applies between connected nodes.

    Features (inspired by EGT - arXiv:2108.03348):
    - Distance bias: applied to ALL node pairs (enables global attention based on graph distance)
    - Edge bias: applied only to CONNECTED nodes (encodes bond features into attention)
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        edge_in_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        expansion_k: int = 4,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.scale = math.sqrt(self.head_dim)

        # Node projections (Q, K, V)
        self.WQ = nn.Linear(node_in_dim, hidden_dim, bias=False)
        self.WK = nn.Linear(node_in_dim, hidden_dim, bias=False)
        self.WV = nn.Linear(node_in_dim, hidden_dim, bias=False)

        # Node output projection
        self.WO = nn.Linear(hidden_dim, node_in_dim, bias=True)

        # Edge bias module (only for connected nodes)
        if edge_in_dim is not None:
            self.WE_logits = nn.Linear(edge_in_dim, num_heads, bias=True)

        # Node norms (pre-attention and pre-FFN)
        self.norm1 = nn.LayerNorm(node_in_dim)
        self.norm2 = nn.LayerNorm(node_in_dim)

        # Dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)

        # Node FFN
        ffn_hidden = expansion_k * node_in_dim
        self.ffn = nn.Sequential(
            LnBnDr(
                node_in_dim,
                ffn_hidden,
                dropout=dropout,
                activation=activation,
                norm=None,
            ),
            LnBnDr(
                ffn_hidden, node_in_dim, dropout=dropout, activation=None, norm=None
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        graph_id: torch.Tensor,
        dist_bias: Optional[torch.Tensor] = None,
        edge_graph_ids: Optional[torch.Tensor] = None,
        local_src: Optional[torch.Tensor] = None,
        local_dst: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the GT convolution layer.

        :param torch.Tensor x: Node features [N, node_in_dim].
        :param torch.Tensor edge_index: Edge indices [2, E].
        :param torch.Tensor | None edge_attr: Edge features [E, edge_in_dim] or None.
        :param torch.Tensor graph_id: Graph assignment for each node [N] (long tensor).
        :param torch.Tensor | None dist_bias: Distance bias for all node pairs
            [B, max_nodes, max_nodes, num_heads], or None.
        :param torch.Tensor | None edge_graph_ids: Optional precomputed graph id
            for each edge [E]. Computed from ``edge_index`` and ``graph_id`` when
            not provided (pass precomputed values to avoid redoing the work per
            layer).
        :param torch.Tensor | None local_src: Optional precomputed local source
            index for each edge [E]. Computed inline when not provided.
        :param torch.Tensor | None local_dst: Optional precomputed local
            destination index for each edge [E]. Computed inline when not
            provided.
        :returns: Updated node features [N, node_in_dim].
        :rtype: torch.Tensor
        """
        if graph_id.dtype != torch.long:
            graph_id = graph_id.long()

        x_res = x

        # Pre-norm for attention
        x_norm = self.norm1(x_res)

        # Project to Q, K, V
        Q = self.WQ(x_norm)  # [N, hidden_dim]
        K = self.WK(x_norm)
        V = self.WV(x_norm)

        # Convert to dense batch for global attention: [N, D] -> [B, max_nodes, D]
        Q_dense, mask = to_dense_batch(Q, graph_id)
        K_dense, _ = to_dense_batch(K, graph_id)
        V_dense, _ = to_dense_batch(V, graph_id)

        B, max_nodes, _ = Q_dense.shape

        # Reshape for multi-head attention: [B, max_nodes, H, Dh] -> [B, H, max_nodes, Dh]
        Q_dense = Q_dense.view(B, max_nodes, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K_dense = K_dense.view(B, max_nodes, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V_dense = V_dense.view(B, max_nodes, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Compute attention logits for ALL node pairs: [B, H, max_nodes, max_nodes]
        attn_logits = torch.matmul(Q_dense, K_dense.transpose(-2, -1)) / self.scale

        # Add distance bias (applies to ALL node pairs - this is the key difference from sparse attention)
        if dist_bias is not None:
            # dist_bias: [B, max_nodes, max_nodes, H] -> [B, H, max_nodes, max_nodes]
            attn_logits = attn_logits + dist_bias.permute(0, 3, 1, 2)

        # Add edge bias at connected node positions only
        if self.edge_in_dim is not None and edge_attr is not None:
            if edge_graph_ids is None or local_src is None or local_dst is None:
                src, dst = edge_index
                edge_graph_ids = graph_id[src]
                num_nodes_per_graph = torch.bincount(graph_id, minlength=B)
                node_offsets = torch.cat(
                    [
                        torch.zeros(1, device=graph_id.device, dtype=torch.long),
                        torch.cumsum(num_nodes_per_graph[:-1], dim=0),
                    ]
                )
                local_src = src - node_offsets[edge_graph_ids]
                local_dst = dst - node_offsets[edge_graph_ids]

            edge_bias = self.WE_logits(edge_attr)  # [E, H]
            attn_logits[
                edge_graph_ids.unsqueeze(1).expand(-1, self.num_heads),
                torch.arange(self.num_heads, device=x.device)
                .unsqueeze(0)
                .expand(edge_attr.size(0), -1),
                local_src.unsqueeze(1).expand(-1, self.num_heads),
                local_dst.unsqueeze(1).expand(-1, self.num_heads),
            ] += edge_bias

        # Safe-softmax: mask keys on padded positions AND zero out logits for
        # fully-padded query rows to avoid NaN gradients through WV / WO.
        key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, max_nodes]
        query_mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, max_nodes, 1]
        attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))
        # For rows where all keys are masked (fully-padded queries), replace
        # the entire row with a finite constant so softmax returns a valid
        # (uniform) distribution instead of NaN.
        attn_logits = attn_logits.masked_fill(~query_mask, 0.0)

        # Softmax and dropout
        attn_weights = torch.softmax(attn_logits, dim=-1)
        # Zero out weights for fully-padded query rows (their outputs get
        # discarded by from_dense_batch anyway).
        attn_weights = attn_weights.masked_fill(~query_mask, 0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values: [B, H, max_nodes, Dh]
        out = torch.matmul(attn_weights, V_dense)

        out = out.transpose(1, 2).reshape(
            B, max_nodes, self.hidden_dim
        )  # [B, max_nodes, hidden_dim]

        # Convert back to sparse format
        out = from_dense_batch(out, graph_id)  # [N, hidden_dim]

        # Output projection + residual
        attn_out = self.WO(out)
        attn_out = self.dropout_layer(attn_out)
        x1 = x_res + attn_out

        # Pre-FFN norm + FFN + residual
        x1_norm = self.norm2(x1)
        ffn_out = self.ffn(x1_norm)
        x_out = x1 + ffn_out

        return x_out


@EncoderRegistry.register()
class GT(BaseGraphEncoder, HyperparametersMixin):
    """Graph Transformer encoder using stacked GTConv layers with distance-biased attention.

    A graph transformer architecture that uses multi-head self-attention
    with edge features and shortest-path distance encoding as attention bias.

    Architecture: atom_projection, bond_projection → GTConv stack (with distance bias) → readout

    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.

    References:
    - Edge-augmented Graph Transformer (EGT): https://arxiv.org/abs/2108.03348
    - GPS: https://arxiv.org/abs/2205.12454

    It is intended to be used inside a :class:`BaseClassicModel` instance.

    :param int num_layers: number of GTConv layers
    :param int atom_input_dim: number of input atom features from GraphFeaturizer
    :param int bond_input_dim: number of input bond features from GraphFeaturizer
    :param int atom_hidden_dim: hidden dimension for atom features in GTConv layers
    :param int num_heads: number of attention heads, must divide atom_hidden_dim evenly
    :param int expansion_k: expansion factor for the feed-forward network in GTConv layers
    :param int | None distance_k: maximum shortest-path distance for spatial encoding, None to disable
    :param str activation: activation function to be used in all layers
    :param float dropout: dropout noise level
    :param str jk: jumping knowledge strategy ('last', 'concat', 'max', 'sum')
    :param str readout: readout function to aggregate atom representations
    """

    def __init__(
        self,
        num_layers: int,
        atom_input_dim: int,
        bond_input_dim: int,
        atom_hidden_dim: int,
        num_heads: int,
        expansion_k: int,
        distance_k: int | None,
        activation: str,
        dropout: float,
        jk: str,
        readout: str,
        laplacian_k: int,
        rwse_k: int,
        elstatic_k: int,
        distmat_k: int,
        rrwp_k: int,
    ):
        super().__init__(laplacian_k, rwse_k, elstatic_k, distmat_k, rrwp_k)
        # Snap num_heads to the largest divisor of atom_hidden_dim that is <= num_heads,
        # so HPO can freely sample both values without causing a shape mismatch.
        requested_num_heads = num_heads
        while num_heads > 1 and atom_hidden_dim % num_heads != 0:
            num_heads -= 1
        if num_heads != requested_num_heads:
            _LOGGER.warning(
                "GT: num_heads=%d does not divide atom_hidden_dim=%d; "
                "snapping to num_heads=%d.",
                requested_num_heads,
                atom_hidden_dim,
                num_heads,
            )

        self.save_hyperparameters()

        # Input projections (similar to GPS)
        self.atom_projection = nn.Sequential(
            LnBnDr(atom_input_dim, atom_hidden_dim, dropout, activation, None),
            LnBnDr(atom_hidden_dim, atom_hidden_dim, dropout, None, None),
        )
        self.bond_projection = nn.Sequential(
            LnBnDr(bond_input_dim, atom_hidden_dim, dropout, activation, None),
            LnBnDr(atom_hidden_dim, atom_hidden_dim, dropout, None, None),
        )

        # Stack of GTConv layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GTConv(
                    node_in_dim=atom_hidden_dim,
                    hidden_dim=atom_hidden_dim,
                    edge_in_dim=atom_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                    expansion_k=expansion_k,
                )
            )

        # Spatial encoder for distance bias (outputs per-head bias)
        if distance_k is not None:
            self.dist_encoder = SpatialEncoder(distance_k, num_heads)
        else:
            self.dist_encoder = None

        self._parse_jk(jk)
        self._parse_readout(readout)

    def _get_distance_bias(self, graph: Batch) -> torch.Tensor | None:
        """Compute distance bias for ALL node pairs from the SPD matrix.

        :param Batch graph: batched PyG graph with spd attribute [batch_size, max_nodes, max_nodes]
        :return torch.Tensor | None: distance bias [batch_size, max_nodes, max_nodes, num_heads] or None
        """
        if self.dist_encoder is None:
            return None
        if not hasattr(graph, "spd") or graph.spd is None:
            _LOGGER.warning(
                "GT: distance_k is set but graph.spd is missing; distance "
                "bias will be skipped. Ensure the datamodule is configured "
                "with compute_distances=True."
            )
            return None

        # Encode distances: [batch_size, max_nodes, max_nodes] -> [batch_size, max_nodes, max_nodes, num_heads]
        return self.dist_encoder(graph.spd)

    def forward(self, graph: Batch) -> torch.Tensor:
        """Converts a batched PyG graph into a (batch_size, fp_dim) tensor.

        :param Batch graph: batched PyG graph from the dataloader
        :return torch.Tensor: learned molecular representation
        """
        g, atom_feats, bond_feats, graph_id = self._process_graph_batch(graph)

        if graph_id.dtype != torch.long:
            graph_id = graph_id.long()

        # Project input features to hidden dimension
        atom_feats = self.atom_projection(atom_feats)
        bond_feats = self.bond_projection(bond_feats)

        # Get distance bias for ALL node pairs
        dist_bias = self._get_distance_bias(g)

        # Precompute per-edge local indices once — these are pure functions of
        # graph_id and edge_index and do not change across layers.
        edge_graph_ids: torch.Tensor | None = None
        local_src: torch.Tensor | None = None
        local_dst: torch.Tensor | None = None
        if bond_feats is not None and g.edge_index is not None:
            src, dst = g.edge_index
            edge_graph_ids = graph_id[src]
            batch_size = int(graph_id.max().item()) + 1 if graph_id.numel() > 0 else 0
            num_nodes_per_graph = torch.bincount(graph_id, minlength=batch_size)
            node_offsets = torch.cat(
                [
                    torch.zeros(1, device=graph_id.device, dtype=torch.long),
                    torch.cumsum(num_nodes_per_graph[:-1], dim=0),
                ]
            )
            local_src = src - node_offsets[edge_graph_ids]
            local_dst = dst - node_offsets[edge_graph_ids]

        all_atom_feats = []

        # Pass through GTConv stack with distance bias
        for layer in self.layers:
            atom_feats = layer(
                atom_feats,
                g.edge_index,
                bond_feats,
                graph_id,
                dist_bias=dist_bias,
                edge_graph_ids=edge_graph_ids,
                local_src=local_src,
                local_dst=local_dst,
            )
            all_atom_feats.append(atom_feats)

        # Apply jumping knowledge
        final_atom_feats = self._run_jk(all_atom_feats)

        return self.readout(g, final_atom_feats)
