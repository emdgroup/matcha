"""E(n) Equivariant Graph Neural Network (E3GNN) encoder for 3D molecular conformers."""

import torch
import torch.nn as nn
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import scatter

from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from matcha.nn.layers import LnBnDr


# Helper functions
def exists(val):
    return val is not None


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    """Fourier-encode scalar distances into multi-scale sinusoidal features.

    :param torch.Tensor x: Distance values to encode.
    :param int num_encodings: Number of frequency scales (powers of 2).
    :param bool include_self: Whether to concatenate the original distance values.
    :returns: Fourier-encoded features.
    :rtype: torch.Tensor
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class SiLU(nn.Module):
    """SiLU activation function."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class CoorsNorm(nn.Module):
    """Coordinate normalization layer with learnable scale.

    :param float eps: Epsilon for numerical stability in norm computation.
    :param float scale_init: Initial value for the learnable scale parameter.
    """

    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class EGNN_Sparse(MessagePassing):
    """E(n) Equivariant Graph Neural Network layer for sparse graphs.

    Separates the edge assignment from the computation, allowing for great
    reduction in time and computations when the graph is locally or sparsely connected.

    Reference: https://arxiv.org/abs/2102.09844

    :param int feats_dim: Dimension of node features
    :param int edge_attr_dim: Dimension of edge attributes
    :param int m_dim: Dimension of message hidden layer
    :param int fourier_features: Number of Fourier features for distance encoding
    :param bool soft_edge: Whether to use soft edge attention
    :param bool norm_feats: Whether to normalize node features
    :param bool norm_coors: Whether to normalize coordinates
    :param float norm_coors_scale_init: Initial scale for coordinate normalization
    :param bool update_feats: Whether to update node features
    :param bool update_coors: Whether to update coordinates
    :param float dropout: Dropout rate
    :param float | None coor_weights_clamp_value: Clamp value for coordinate weights
    :param str aggr: Feature aggregation method ('add', 'mean', 'max')
    :param str coord_aggr: Coordinate-update aggregation method ('add', 'mean', 'max').
        Decoupled from ``aggr`` because the reference paper (Satorras et al.)
        aggregates coord updates with a mean while feature updates can use sum.
    """

    def __init__(
        self,
        feats_dim: int,
        edge_attr_dim: int = 0,
        m_dim: int = 16,
        fourier_features: int = 0,
        soft_edge: bool = False,
        norm_feats: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        update_feats: bool = True,
        update_coors: bool = True,
        dropout: float = 0.0,
        coor_weights_clamp_value: float | None = None,
        aggr: str = "add",
        coord_aggr: str = "mean",
        **kwargs,
    ):
        assert aggr in {"add", "sum", "max", "mean"}, (
            "Aggregation must be a valid option"
        )
        assert coord_aggr in {"add", "sum", "max", "mean"}, (
            "Coordinate aggregation must be a valid option"
        )
        assert update_feats or update_coors, (
            "Must update either features, coordinates, or both"
        )
        kwargs.setdefault("aggr", aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)

        # Model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.coord_aggr = coord_aggr

        self.edge_input_dim = (
            (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        )
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout_layer,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU(),
        )

        # Soft edge attention
        self.edge_weight = (
            nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edge else None
        )

        # Node normalization - use PyG LayerNorm for graph data
        self.node_norm = torch.nn.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )

        # Node MLP
        self.node_mlp = (
            nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 2),
                self.dropout_layer,
                SiLU(),
                nn.Linear(feats_dim * 2, feats_dim),
            )
            if update_feats
            else None
        )

        # Coordinate MLP
        self.coors_mlp = (
            nn.Sequential(
                nn.Linear(m_dim, m_dim * 4),
                self.dropout_layer,
                SiLU(),
                nn.Linear(m_dim * 4, 1),
            )
            if update_coors
            else None
        )

        self.apply(self._init_weights)

        # Re-init the final coord-MLP linear with a small gain so the coord
        # update starts near-identity, matching the reference E_GCL implementation.
        # (Without this, default-Xavier init on this output is a known instability
        # in `EGNN_Sparse`.)
        if self.coors_mlp is not None:
            nn.init.xavier_uniform_(self.coors_mlp[-1].weight, gain=1e-3)
            nn.init.zeros_(self.coors_mlp[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        coors: Tensor,
        feats: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        batch: Adj = None,
        size: Size = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        :param Tensor coors: Node coordinates [num_nodes, pos_dim]
        :param Tensor feats: Node features [num_nodes, feats_dim]
        :param Adj edge_index: Edge indices [2, num_edges]
        :param OptTensor edge_attr: Edge attributes [num_edges, edge_attr_dim]
        :param Adj batch: Batch assignment [num_nodes]
        :param Size size: Size for bipartite graphs
        :return tuple[Tensor, Tensor]: Updated coordinates [num_nodes, pos_dim] and
            features [num_nodes, feats_dim] as separate tensors.
        """
        rel_coors = coors[edge_index[1]] - coors[edge_index[0]]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(
                rel_dist, num_encodings=self.fourier_features
            )
            rel_dist = rel_dist.squeeze(1)  # n () d -> n d

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(
            edge_index,
            x=feats,
            edge_attr=edge_attr_feats,
            coors=coors,
            rel_coors=rel_coors,
            batch=batch,
        )
        return coors_out, hidden_out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """Create messages from neighboring nodes."""
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """Propagate messages with coordinate updates."""
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        update_kwargs = self.inspector.collect_param_data("update", coll_dict)

        # Get messages
        m_ij = self.message(**msg_kwargs)

        # Update coordinates if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # Clamp if arg is set
            if self.coor_weights_clamp_value is not None:
                coor_wij = coor_wij.clamp(
                    min=-self.coor_weights_clamp_value,
                    max=self.coor_weights_clamp_value,
                )

            # Normalize if needed
            rel_coors = self.coors_norm(kwargs["rel_coors"])

            # Coord aggregation is decoupled from feature aggregation:
            # the paper (Satorras et al.) uses mean here while feature updates
            # commonly use sum. Bypass `self.aggregate` (which is tied to
            # `self.aggr`) and reduce with the configured `coord_aggr` instead.
            mhat_i = scatter(
                coor_wij * rel_coors,
                aggr_kwargs["index"],
                dim=0,
                dim_size=aggr_kwargs.get("dim_size"),
                reduce=self.coord_aggr,
            )
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # Update features if specified
        if self.update_feats:
            # Weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = (
                self.node_norm(kwargs["x"]) if self.node_norm else kwargs["x"]
            )
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = kwargs["x"] + hidden_out
        else:
            hidden_out = kwargs["x"]

        # Return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)


@EncoderRegistry.register()
class E3GNN(BaseGraphEncoder, HyperparametersMixin):
    """E(n) Equivariant Graph Neural Network encoder.

    This class performs message passing on molecular conformers using E(n) equivariant
    convolutions, updates the atom representations according to relative atomic
    positions and bond information, then returns a molecular embedding for property
    prediction.

    Uses the EGNN_Sparse layer for efficient message passing on sparse graphs.

    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.

    References:
    - https://arxiv.org/abs/2102.09844 (E(n) Equivariant Graph Neural Networks)

    It is intended to be used inside a :class:`BaseClassicModel` instance.

    :param int num_layers: number of message passing layers
    :param int atom_input_dim: number of input atom features from GraphFeaturizer
    :param int bond_input_dim: number of input bond features from GraphFeaturizer
    :param int hidden_features: number of hidden features in message passing layers
    :param int m_dim: dimension of message hidden layer in EGNN
    :param int fourier_features: number of Fourier features for distance encoding
    :param bool soft_edge: whether to use soft edge attention
    :param bool norm_feats: whether to normalize node features
    :param bool norm_coors: whether to normalize coordinates
    :param bool update_coors: whether to update coordinates during message passing
    :param str activation: activation function to be used in projection layers
    :param float dropout: dropout noise level for projection layers
    :param float coor_weights_clamp_value: symmetric clamp on the per-edge
        coordinate weight before aggregation. Matches the paper's
        ``torch.clamp(min=-100, max=100)`` behaviour when set to ``100.0``.
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
        m_dim: int,
        fourier_features: int,
        soft_edge: bool,
        norm_feats: bool,
        norm_coors: bool,
        update_coors: bool,
        activation: str,
        dropout: float,
        coor_weights_clamp_value: float,
        jk: str,
        readout: str,
        laplacian_k: int,
        rwse_k: int,
        elstatic_k: int,
        distmat_k: int,
        rrwp_k: int,
    ):
        super().__init__(
            laplacian_k,
            rwse_k,
            elstatic_k,
            distmat_k,
            rrwp_k,
        )
        self.save_hyperparameters()

        # Input projection: atom features -> hidden features
        self.atom_projection = nn.Sequential(
            LnBnDr(atom_input_dim, atom_hidden_dim, dropout, activation, "batch"),
            LnBnDr(atom_hidden_dim, atom_hidden_dim, dropout, None, None),
        )

        # Message passing layers (EGNN_Sparse)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                EGNN_Sparse(
                    feats_dim=atom_hidden_dim,
                    edge_attr_dim=bond_input_dim,
                    m_dim=m_dim,
                    fourier_features=fourier_features,
                    soft_edge=soft_edge,
                    norm_feats=norm_feats,
                    norm_coors=norm_coors,
                    norm_coors_scale_init=1e-2,
                    update_feats=True,
                    update_coors=update_coors,
                    dropout=dropout,
                    coor_weights_clamp_value=coor_weights_clamp_value,
                    aggr="add",
                    coord_aggr="mean",
                )
            )

        self._parse_jk(jk)
        self._parse_readout(readout)

    @property
    def fp_dim(self) -> int:
        """Return fingerprint dimension."""
        return self._fp_dim

    def forward(self, graph: Batch, coords: torch.Tensor) -> torch.Tensor:
        """Converts a batched PyG graph and 3D coordinates into a learned representation.

        :param Batch graph: batched PyG graph from the dataloader
        :param torch.Tensor coords: batched 3D coordinates [num_atoms, 3] from the dataloader
        :return torch.Tensor: learned molecular representation [batch_size, fp_dim]
        """
        g, atom_feats, bond_feats, graph_id = self._process_graph_batch(graph)

        # Project atom features to hidden space
        atom_feats = self.atom_projection(atom_feats)

        # Message passing
        feats = atom_feats
        all_atom_feats = []
        for layer in self.layers:
            coords, feats = layer(
                coords, feats, g.edge_index, edge_attr=bond_feats, batch=graph_id
            )
            all_atom_feats.append(feats)

        # Apply jumping knowledge
        final_atom_feats = self._run_jk(all_atom_feats)

        return self.readout(g, final_atom_feats)
