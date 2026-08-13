"""E(n) Equivariant Graph Neural Network (E3GNN) encoder for 3D molecular conformers."""

from typing import Optional

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


def fourier_encode_dist(squared_dist, num_encodings=4, include_self=True):
    """Fourier-encode squared distances into multi-scale sinusoidal features.

    Callers pass the per-edge squared distance ``(r_j - r_i).pow(2).sum(-1)``;
    no square root is taken inside this function.

    :param torch.Tensor squared_dist: Squared distance values to encode.
    :param int num_encodings: Number of frequency scales (powers of 2).
    :param bool include_self: Whether to concatenate the original squared-distance values.
    :returns: Fourier-encoded features.
    :rtype: torch.Tensor
    """
    x = squared_dist.unsqueeze(-1)
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
        # Cache aggr strings for our overrides of `aggregate()` -- avoids
        # depending on the PyG-normalized `self.aggr` (which may be an
        # `Aggregation` object rather than a string in newer PyG versions).
        self.feat_aggr = aggr
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
        )
        return coors_out, hidden_out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Tensor,
        rel_coors: Tensor,
    ) -> Tensor:
        """Compute per-edge messages and coord deltas, packed into one tensor.

        The last three columns hold the per-edge weighted coord delta; the
        preceding ``m_dim`` columns hold the feature message (already
        soft-edge weighted when ``self.soft_edge`` is enabled). :meth:`aggregate`
        splits them and reduces each half with its own aggregation.
        """
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            if self.coor_weights_clamp_value is not None:
                coor_wij = coor_wij.clamp(
                    min=-self.coor_weights_clamp_value,
                    max=self.coor_weights_clamp_value,
                )
            coord_delta = coor_wij * self.coors_norm(rel_coors)
        else:
            coord_delta = m_ij.new_zeros((m_ij.size(0), 3))

        if self.soft_edge and self.update_feats:
            m_ij = m_ij * self.edge_weight(m_ij)

        return torch.cat([m_ij, coord_delta], dim=-1)

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: OptTensor = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """Reduce feature messages with ``feat_aggr`` and coord deltas with ``coord_aggr``.

        Decoupled per-half reduction is why we need this override -- PyG's
        default aggregate applies a single reduction to the whole tensor.
        """
        m_part = inputs[..., : self.m_dim]
        coord_part = inputs[..., self.m_dim :]
        m_i = scatter(m_part, index, dim=0, dim_size=dim_size, reduce=self.feat_aggr)
        mhat_i = scatter(
            coord_part, index, dim=0, dim_size=dim_size, reduce=self.coord_aggr
        )
        return torch.cat([m_i, mhat_i], dim=-1)

    def update(
        self, aggr_out: Tensor, x: Tensor, coors: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Split the aggregated output into feature and coord halves and finalize."""
        m_i = aggr_out[..., : self.m_dim]
        mhat_i = aggr_out[..., self.m_dim :]

        coors_out = coors + mhat_i if self.update_coors else coors

        if self.update_feats:
            hidden_feats = self.node_norm(x) if self.node_norm else x
            hidden_out = x + self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        else:
            hidden_out = x

        return hidden_out, coors_out


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
    :param float norm_coors_scale_init: initial value of the learnable scale
        parameter inside :class:`CoorsNorm` when ``norm_coors=True``. Only
        used when coordinate normalization is enabled.
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
        norm_coors_scale_init: float,
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

        # Input projection: atom features -> hidden features.
        # Second stage is a plain Linear+Dropout -- writing it as
        # LnBnDr(dim, dim, dropout, None, None) previously suggested there was
        # a norm/activation involved when there wasn't.
        self.atom_projection = nn.Sequential(
            LnBnDr(atom_input_dim, atom_hidden_dim, dropout, activation, "batch"),
            nn.Linear(atom_hidden_dim, atom_hidden_dim),
            nn.Dropout(dropout),
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
                    norm_coors_scale_init=norm_coors_scale_init,
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

    def forward_nodes_per_layer(self, graph: Batch) -> tuple[list[torch.Tensor], Batch]:
        """Run E(n)-equivariant message passing and return one node-feature
        tensor per layer.

        Coordinates are read from ``graph.pos`` (the PyG convention). The
        3D datamodules (:class:`Graph3DDataModule` and the pretraining
        counterpart) are responsible for attaching them there before the
        batch reaches the encoder.

        :param Batch graph: Batched PyG graph from the dataloader. Must
            expose per-node 3D coordinates on ``graph.pos``.
        :returns: Tuple ``(all_atom_feats, g)`` — ``all_atom_feats`` has
            length ``num_layers``; each entry has shape
            ``[num_nodes, atom_hidden_dim]``.
        :rtype: tuple[list[torch.Tensor], Batch]
        :raises ValueError: If ``graph.pos`` is not set.
        """
        coords = getattr(graph, "pos", None)
        if coords is None:
            raise ValueError(
                "E3GNN requires 3D coordinates on graph.pos, but the batch "
                "has none. Attach per-node coordinates to Data.pos in the "
                "datamodule (see Graph3DDataModule) before calling the encoder."
            )
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

        return all_atom_feats, g
