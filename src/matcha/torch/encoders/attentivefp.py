"""AttentiveFP graph encoder with gated attention mechanism."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder


class GATEConv(MessagePassing):
    """Graph Attention Edge Convolution layer for incorporating edge features into
    node representations using attention mechanism.

    This layer is adapted from the PyTorch Geometric AttentiveFP implementation.
    Reference: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html

    This class is not meant to be used directly, but acts as a building block for
    :class:`AttentiveFP`.

    :param int in_channels: Input node feature dimension.
    :param int out_channels: Output node feature dimension.
    :param int edge_dim: Edge feature dimension.
    :param float dropout: Dropout rate for attention weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        """Forward pass computing attention-weighted message passing.

        :param Tensor x: Node features [num_nodes, in_channels].
        :param Adj edge_index: Edge indices [2, num_edges].
        :param Tensor edge_attr: Edge features [num_edges, edge_dim].
        :returns: Updated node features [num_nodes, out_channels].
        :rtype: Tensor
        """
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        # propagate_type: (x: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        x_j = F.leaky_relu(self.lin1(torch.cat([x_j, edge_attr], dim=-1)), 0.01)
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, 0.01)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)


@EncoderRegistry.register()
class AttentiveFP(BaseGraphEncoder, HyperparametersMixin):
    """Attention-based graph encoder as described in 'Pushing the Boundaries of Molecular
    Representation for Drug Discovery with the Graph Attention Mechanism'
    This class performs message passing, updates the atom representations and
    returns a molecular embedding which can then be used for molecular property
    prediction.
    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.
    References:
    - https://www.ncbi.nlm.nih.gov/pubmed/31408336
    - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html

    It is intended to be used inside a :class:`BaseClassicModel` instance.
    Check the docs of :class:`matcha.torch.models.classic.AttentiveFPModel` for further details.

    :param int num_layers: number of message passing layers

    :param int atom_input_dim: number of input atom features from GraphFeaturizer

    :param int bond_input_dim: number of input bond features from GraphFeaturizer

    :param int atom_hidden_dim: number of hidden atom features in message passing layers

    :param float dropout: dropout noise level

    :param str readout: readout function to aggregate all atom representations

    :param str jk: jumping knowledge strategy to use when returning molecular
        representations after forward pass
    """

    def __init__(
        self,
        num_layers: int,
        atom_input_dim: int,
        bond_input_dim: int,
        atom_hidden_dim: int,
        dropout: float,
        readout: str,
        jk: str,
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
        self._parse_jk(jk)
        self._parse_readout(readout)
        self.dropout = dropout

        # Initial atom embedding layer
        self.atom_projection = nn.Linear(atom_input_dim, atom_hidden_dim)

        # Initial context layer with edge features (GATEConv + GRU)
        self.gate_conv = GATEConv(
            atom_hidden_dim, atom_hidden_dim, bond_input_dim, dropout
        )
        self.gru = GRUCell(atom_hidden_dim, atom_hidden_dim)

        # Subsequent GNN layers (GATConv + GRU)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATv2Conv(
                atom_hidden_dim,
                atom_hidden_dim,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            gru = GRUCell(atom_hidden_dim, atom_hidden_dim)
            self.layers.append(nn.ModuleList([conv, gru]))

    def forward_nodes_per_layer(self, graph: Batch) -> tuple[list[torch.Tensor], Batch]:
        """Run AttentiveFP message passing and return one node-feature tensor per layer.

        The initial GATEConv+GRU context layer produces the first entry; each of
        the subsequent ``num_layers - 1`` GATv2Conv+GRU layers appends another,
        so the returned list has length ``num_layers``.

        :param Batch graph: Batched PyG graph from the dataloader.
        :returns: Tuple ``(all_atom_feats, g)`` — ``all_atom_feats`` has length
            ``num_layers``; each entry has shape ``[num_nodes, atom_hidden_dim]``.
        :rtype: tuple[list[torch.Tensor], Batch]
        """
        g, atom_feats, bond_feats, _ = self._process_graph_batch(graph)
        all_atom_feats = []

        # Initial atom embedding
        x = F.leaky_relu_(self.atom_projection(atom_feats), 0.01)

        # Initial context with edge features
        h = F.elu_(self.gate_conv(x, g.edge_index, bond_feats))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()
        all_atom_feats.append(x)

        # Subsequent GNN layers
        for conv, gru in self.layers:
            h = conv(x, g.edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
            if all_atom_feats != []:
                x = x + all_atom_feats[-1]
            all_atom_feats.append(x)

        return all_atom_feats, g
