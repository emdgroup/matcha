"""General, Powerful, Scalable (GPS) Graph Transformer encoder."""

import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import aggr
from torch_geometric.utils import to_dense_batch

from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from matcha.nn.activations import ActivationRegistry
from matcha.nn.layers import (
    LayerRegistry,
    LnBnDr,
    SpatialEncoder,
    BiasedMultiHeadAttention,
    from_dense_batch,
)
from matcha.utils.logging import get_default_logger

_LOGGER = get_default_logger(__name__)


class MPNNPlusConv(nn.Module):
    """Minimal reimplementation of MPNNPlus convolution layer for GPS.

    GPS++ style message passing layer that updates both node and edge features.
    Uses variance-preserving aggregation for better gradient flow.
    No internal normalization - relies on pre-norm pattern in GPSBlock. Returns
    pure deltas (no internal residual) so that GPSBlock can provide a single,
    symmetric skip connection over both the message-passing and attention
    branches.

    Reference: https://arxiv.org/abs/2212.02229

    :param int in_dim: Input node feature dimension
    :param int out_dim: Output node feature dimension
    :param int in_dim_edges: Input edge feature dimension
    :param int out_dim_edges: Output edge feature dimension
    :param str activation: Activation function name
    :param float dropout: Node dropout rate
    :param float edge_dropout: Edge dropout rate
    :param int mlp_expansion_ratio: Hidden dim expansion factor
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        out_dim_edges: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        edge_dropout: float = 0.0,
        mlp_expansion_ratio: int = 4,
    ):
        super().__init__()

        # Variance-preserving aggregation from PyG
        self.aggregator = aggr.VariancePreservingAggregation()

        # Edge model: takes [sender_feat, receiver_feat, edge_feat] -> updated edge_feat
        # Using concat mode: 2 * in_dim + in_dim_edges
        edge_model_in_dim = 2 * in_dim + in_dim_edges
        edge_model_hidden_dim = mlp_expansion_ratio * in_dim_edges
        self.edge_model = nn.Sequential(
            LnBnDr(
                edge_model_in_dim,
                edge_model_hidden_dim,
                dropout=edge_dropout,
                activation=activation,
                norm=None,
            ),
            LnBnDr(
                edge_model_hidden_dim,
                out_dim_edges,
                dropout=edge_dropout,
                activation=None,
                norm=None,
            ),
        )

        # Node model: takes [aggregated_messages, node_feat] -> updated node_feat
        # VPA outputs same dim as input, scatter to both: 3 * in_dim + 2 * edge_dim
        node_model_in_dim = 3 * in_dim + 2 * out_dim_edges
        node_model_hidden_dim = mlp_expansion_ratio * in_dim
        self.node_model = nn.Sequential(
            LnBnDr(
                node_model_in_dim,
                node_model_hidden_dim,
                dropout=dropout,
                activation=activation,
                norm=None,
            ),
            LnBnDr(
                node_model_hidden_dim,
                out_dim,
                dropout=dropout,
                activation=None,
                norm=None,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param torch.Tensor x: Node features [num_nodes, in_dim]
        :param torch.Tensor edge_index: Edge indices [2, num_edges]
        :param torch.Tensor edge_attr: Edge features [num_edges, in_dim_edges]
        :return tuple: Node and edge deltas (no internal residual)
        """
        senders = edge_index[0]
        receivers = edge_index[1]
        num_nodes = x.size(0)

        # Gather node features for edges
        sender_feats = x[senders]  # [num_edges, in_dim]
        receiver_feats = x[receivers]  # [num_edges, in_dim]

        # ===== Edge update =====
        # Concatenate sender, receiver, and edge features
        edge_input = torch.cat([sender_feats, receiver_feats, edge_attr], dim=-1)
        edge_attr_new = self.edge_model(edge_input)

        # ===== Node update =====
        # Create messages by concatenating edge features with node features
        # Scatter to receivers: message = [edge_feat, sender_feat]
        msg_to_receivers = torch.cat([edge_attr_new, sender_feats], dim=-1)
        agg_to_receivers = self.aggregator(
            msg_to_receivers, receivers, dim_size=num_nodes
        )

        # Scatter to senders: message = [edge_feat, receiver_feat]
        msg_to_senders = torch.cat([edge_attr_new, receiver_feats], dim=-1)
        agg_to_senders = self.aggregator(msg_to_senders, senders, dim_size=num_nodes)

        # Combine aggregated messages with original node features
        node_input = torch.cat([agg_to_receivers, agg_to_senders, x], dim=-1)
        x_new = self.node_model(node_input)

        return x_new, edge_attr_new


class GPSBlock(nn.Module):
    """General, Powerful, Scalable (GPS) Graph Transformer Block.

    Combines local message passing (MPNNPlus) with global self-attention.
    Uses PyTorch Geometric for graph operations.

    References:
    - GPS: https://arxiv.org/abs/2205.12454
    - GPS++: https://arxiv.org/abs/2212.02229

    :param int atom_feats: Node feature dimension
    :param int edge_feats: Edge feature dimension
    :param float dropout: Dropout ratio
    :param str norm: Normalization type
    :param str activation: Activation function
    :param int num_heads: Number of attention heads
    :param int expansion_k: FFN expansion factor
    :param int distance_k: Maximum distance for spatial encoding (unused, kept for API compatibility)
    """

    def __init__(
        self,
        atom_feats: int,
        edge_feats: int,
        dropout: float = 0.0,
        norm: str = "adarmsn",
        activation: str = "swish",
        num_heads: int = 4,
        expansion_k: int = 2,
        distance_k: int | None = 5,
    ):
        super().__init__()

        # Local message passing layer (MPNNPlus style with VPA)
        self.mp = MPNNPlusConv(
            in_dim=atom_feats,
            out_dim=atom_feats,
            in_dim_edges=edge_feats,
            out_dim_edges=edge_feats,
            activation=activation,
            dropout=dropout,
            edge_dropout=0.0,
            mlp_expansion_ratio=expansion_k,
        )

        # Global attention layer
        self.att = BiasedMultiHeadAttention(atom_feats, num_heads, dropout)

        # Layer norms - pre-norm pattern
        self.norm1_local = LayerRegistry[norm](atom_feats)
        self.norm1_edge = LayerRegistry[norm](edge_feats)
        self.norm1_global = LayerRegistry[norm](atom_feats)
        self.norm2 = LayerRegistry[norm](atom_feats)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(atom_feats, int(atom_feats * expansion_k)),
            ActivationRegistry[activation](),
            nn.Dropout(dropout),
            nn.Linear(int(atom_feats * expansion_k), atom_feats),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        graph: Batch,
        feat: torch.Tensor,
        edge_feat: torch.Tensor,
        graph_id: torch.Tensor,
        dist_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param Batch graph: PyG batched graph
        :param torch.Tensor feat: Node features [num_nodes, atom_feats]
        :param torch.Tensor edge_feat: Edge features [num_edges, edge_feats]
        :param torch.Tensor graph_id: Batch assignment [num_nodes]
        :param torch.Tensor dist_bias: Distance bias for attention or None
        :return tuple: Updated node and edge features
        """
        # Store inputs for residuals
        h_in = feat
        edge_in = edge_feat

        # Local message passing branch (pre-norm nodes + edges)
        h_local_in = self.norm1_local(h_in)
        edge_local_in = self.norm1_edge(edge_in)
        h_local_delta, edge_delta = self.mp(h_local_in, graph.edge_index, edge_local_in)

        # Global attention branch (pre-norm nodes)
        h_global = self.norm1_global(h_in)
        h_global_dense, mask = to_dense_batch(h_global, graph_id)

        # Apply attention
        h_global_dense = self.att(h_global_dense, attn_bias=dist_bias, attn_mask=mask)
        h_global_delta = from_dense_batch(h_global_dense, graph_id)

        # Combine both branches with single symmetric skip over pre-normed inputs
        h = h_in + h_local_delta + h_global_delta
        edge_feat = edge_in + edge_delta

        # Feed-forward with residual
        h_ffn = self.norm2(h)
        h_ffn = self.mlp(h_ffn)
        h = h + h_ffn

        return h, edge_feat


@EncoderRegistry.register()
class GPS(BaseGraphEncoder, HyperparametersMixin):
    """General, Powerful, Scalable (GPS) Graph Transformer encoder.

    Combines local message passing with global self-attention for learning
    molecular representations. Uses PyTorch Geometric for all graph operations.

    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.

    References:
    - https://arxiv.org/abs/2404.11568
    - https://arxiv.org/abs/2205.12454
    - https://proceedings.mlr.press/v202/ma23c.html

    It is intended to be used inside a :class:`BaseClassicModel` instance.
    Check the docs of :class:`matcha.torch.models.classic.GPSModel` for further details.

    :param int num_layers: number of message passing layers
    :param int atom_input_dim: number of input atom features from GraphFeaturizer
    :param int bond_input_dim: number of input bond features from GraphFeaturizer
    :param int atom_hidden_dim: number of hidden atom (and bond) features in message passing layers
    :param int num_heads: number of attention heads, must divide atom_hidden_dim evenly
    :param int expansion_k: expansion factor for the feed-forward network in GPSBlock
    :param int | None distance_k: Upper bound for the shortest path distance to encode
    :param str activation: activation function to be used in all layers
    :param float dropout: dropout noise level
    :param str | None norm: which norm to use inside LnBnDr layers
    :param str jk: jumping knowledge strategy to use when returning molecular
        representations after forward pass
    :param str readout: readout function to aggregate all atom representations
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
        norm: str | None,
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
                "GPS: num_heads=%d does not divide atom_hidden_dim=%d; "
                "snapping to num_heads=%d.",
                requested_num_heads,
                atom_hidden_dim,
                num_heads,
            )
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GPSBlock(
                    atom_hidden_dim,
                    atom_hidden_dim,
                    dropout,
                    norm,
                    activation,
                    num_heads,
                    expansion_k,
                    distance_k,
                )
            )

        self.atom_projection = nn.Sequential(
            LnBnDr(atom_input_dim, atom_hidden_dim, dropout, activation, norm),
            LnBnDr(atom_hidden_dim, atom_hidden_dim, dropout, None, None),
        )
        self.bond_projection = nn.Sequential(
            LnBnDr(bond_input_dim, atom_hidden_dim, dropout, activation, norm),
            LnBnDr(atom_hidden_dim, atom_hidden_dim, dropout, None, None),
        )

        if distance_k is not None:
            self.dist_encoder = SpatialEncoder(distance_k, num_heads)
        else:
            self.dist_encoder = None

        self._parse_jk(jk)
        self._parse_readout(readout)

    def forward_nodes_per_layer(self, graph: Batch) -> tuple[list[torch.Tensor], Batch]:
        """Run GPS message passing and return one node-feature tensor per layer.

        :param Batch graph: Batched PyG graph from the dataloader.
        :returns: Tuple ``(all_atom_feats, g)`` — ``all_atom_feats`` has length
            ``num_layers``; each entry has shape ``[num_nodes, atom_hidden_dim]``.
        :rtype: tuple[list[torch.Tensor], Batch]
        """
        g, atom_feats, bond_feats, graph_id = self._process_graph_batch(graph)
        all_atom_feats = []
        atom_feats = self.atom_projection(atom_feats)
        bond_feats = self.bond_projection(bond_feats)

        if self.dist_encoder is not None and hasattr(g, "spd") and g.spd is not None:
            dist_bias = self.dist_encoder(g.spd)
        else:
            if self.dist_encoder is not None:
                _LOGGER.warning(
                    "GPS: distance_k is set but graph.spd is missing; "
                    "distance bias will be skipped. Ensure the datamodule "
                    "is configured with compute_distances=True."
                )
            dist_bias = None

        for layer in self.layers:
            atom_feats, bond_feats = layer(
                g, atom_feats, bond_feats, graph_id, dist_bias
            )
            all_atom_feats.append(atom_feats)

        return all_atom_feats, g
