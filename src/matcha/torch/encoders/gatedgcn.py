"""Gated Graph Convolution Network (GatedGCN) encoder."""

import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import nn
from torch.nn import ModuleList
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from matcha.nn.activations import ActivationRegistry
from matcha.nn.layers import LayerRegistry


class GatedGCNConv(MessagePassing):
    """Reimplementation of Residual Gated Graph Convolution, based on
    Graphium's implementation (https://github.com/datamol-io/graphium/blob/f1c1809387184114a06ac2077bb57501bf09f495/graphium/nn/pyg_layers/gated_gcn_pyg.py#L28).

    Original publication is 'Residual Gated Graph ConvNets' (Bresson and Laurent, ICLR 2018).
    Reference: https://arxiv.org/pdf/1711.07553v2.pdf

    :param int in_channels: Input feature dimensions of nodes
    :param int out_channels: Output feature dimensions of nodes
    :param int edge_dim: Input edge-feature dimensions
    :param int out_edge_dim: Output edge-feature dimensions (defaults to edge_dim)
    :param str activation: Activation function name
    :param float dropout: Dropout ratio
    :param str | None norm: Normalization type ('batch', 'layer', 'graph', or None)
    :param float eps: Epsilon for numerical stability in gate normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        out_edge_dim: int | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str | None = None,
        eps: float = 1e-5,
    ):
        super().__init__(aggr="add", flow="source_to_target", node_dim=-2)

        if out_edge_dim is None:
            out_edge_dim = edge_dim

        self.eps = eps
        self.norm_type = norm

        # Linear layers for gating mechanism
        self.A = nn.Linear(in_channels, out_channels, bias=True)
        self.B = nn.Linear(in_channels, out_channels, bias=True)
        self.C = nn.Linear(edge_dim, out_channels, bias=True)
        self.D = nn.Linear(in_channels, out_channels, bias=True)
        self.E = nn.Linear(in_channels, out_channels, bias=True)

        # Edge output projection
        self.edge_out = nn.Sequential(
            nn.Linear(out_channels, out_edge_dim, bias=True),
            nn.Dropout(dropout),
        )

        # Activation and dropout
        self.activation = ActivationRegistry[activation]() if activation else None
        self.dropout = nn.Dropout(dropout)

        # Normalization (handle "graph" separately since it needs batch index)
        if norm is not None and norm != "none":
            self.norm = LayerRegistry[norm](out_channels)
        else:
            self.norm = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the gated graph convolution layer.

        :param torch.Tensor x: Node features with shape [num_nodes, in_channels]
        :param torch.Tensor edge_index: Edge indices with shape [2, num_edges]
        :param torch.Tensor edge_attr: Edge features with shape [num_edges, edge_dim]
        :param torch.Tensor | None batch: Batch assignment for nodes (required for GraphNorm)
        :return tuple[torch.Tensor, torch.Tensor]: Updated node and edge features
        """
        # Apply linear transformations
        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(edge_attr)
        Dx = self.D(x)
        Ex = self.E(x)

        # Propagate messages
        x_out, e_out = self.propagate(
            edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=edge_attr, Ax=Ax
        )

        # Apply normalization, activation, and dropout
        if self.norm is not None:
            if self.norm_type == "graph":
                x_out = self.norm(x_out, batch)
            else:
                x_out = self.norm(x_out)

        if self.activation is not None:
            x_out = self.activation(x_out)

        x_out = self.dropout(x_out)

        # Project edge features
        e_out = self.edge_out(e_out)

        return x_out, e_out

    def message(
        self, Dx_i: torch.Tensor, Ex_j: torch.Tensor, Ce: torch.Tensor
    ) -> torch.Tensor:
        """Compute gated messages.

        :param torch.Tensor Dx_i: Transformed features of target nodes [num_edges, out_channels]
        :param torch.Tensor Ex_j: Transformed features of source nodes [num_edges, out_channels]
        :param torch.Tensor Ce: Transformed edge features [num_edges, out_channels]
        :return torch.Tensor: Gated message weights (sigmoid of sum)
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        # Store edge features for update step
        self._e = e_ij
        return sigma_ij

    def aggregate(
        self,
        sigma_ij: torch.Tensor,
        index: torch.Tensor,
        Bx_j: torch.Tensor,
        Bx: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate gated messages with normalization.

        :param torch.Tensor sigma_ij: Gate weights [num_edges, out_channels]
        :param torch.Tensor index: Target node indices [num_edges]
        :param torch.Tensor Bx_j: Transformed source node features [num_edges, out_channels]
        :param torch.Tensor Bx: Transformed node features [num_nodes, out_channels]
        :return torch.Tensor: Aggregated messages [num_nodes, out_channels]
        """
        dim_size = Bx.shape[0]

        # Weighted sum of messages and sum of gates
        numerator = scatter(
            sigma_ij * Bx_j, index, dim=0, dim_size=dim_size, reduce="sum"
        )
        denominator = scatter(sigma_ij, index, dim=0, dim_size=dim_size, reduce="sum")

        # Handle float16 precision
        dtype = denominator.dtype
        if dtype == torch.float16:
            numerator = numerator.to(dtype=torch.float32)
            denominator = denominator.to(dtype=torch.float32)

        # Normalize by sum of gates
        out = numerator / (denominator + self.eps)

        if dtype == torch.float16:
            out = out.to(dtype=dtype)

        return out

    def update(
        self, aggr_out: torch.Tensor, Ax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update node features with residual connection.

        :param torch.Tensor aggr_out: Aggregated messages [num_nodes, out_channels]
        :param torch.Tensor Ax: Transformed node features [num_nodes, out_channels]
        :return tuple[torch.Tensor, torch.Tensor]: Updated node and edge features
        """
        x = Ax + aggr_out
        e_out = self._e
        del self._e
        return x, e_out


@EncoderRegistry.register()
class GatedGCN(BaseGraphEncoder, HyperparametersMixin):
    """Gated Graph Convolution Network (GatedGCN) encoder as described in 'Benchmarking
    Graph Neural Networks'. This class performs message passing via atom convolutions,
    residual connections and edge convolutions, updates the atom representations
    and returns a molecular embedding which can then be used for molecular
    property prediction.

    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.

    References:
    - https://arxiv.org/abs/2003.00982
    - https://pytorch-geometric.readthedocs.io/en/latest/

    It is intended to be used inside a :class:`BaseClassicModel` instance.
    Check the docs of :class:`matcha.torch.models.classic.GatedGCNModel` for further details.

    :param int num_layers: number of message passing layers

    :param int atom_input_dim: number of input atom features from GraphFeaturizer

    :param int bond_input_dim: number of input bond features from GraphFeaturizer

    :param int atom_hidden_dim: number of hidden atom (and bond) features in message passing layers

    :param str activation: activation function to be used in all layers

    :param float dropout: dropout noise level

    :param str norm: which norm to use inside GatedGCN layers

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
        self.save_hyperparameters()
        self._parse_jk(jk)
        self._parse_readout(readout)

        self.layers = ModuleList()

        for i in range(num_layers):
            in_channels = atom_input_dim if i == 0 else atom_hidden_dim
            edge_dim = bond_input_dim if i == 0 else atom_hidden_dim
            self.layers.append(
                GatedGCNConv(
                    in_channels=in_channels,
                    out_channels=atom_hidden_dim,
                    edge_dim=edge_dim,
                    out_edge_dim=atom_hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                )
            )

    def forward(self, graph: Batch) -> torch.Tensor:
        """Converts a batched PyG graph into a (batch_size, fp_dim) tensor for further
        processing.

        :param Batch graph: batched PyG graph from the dataloader

        :return torch.Tensor: learned representation
        """
        g, atom_feats, bond_feats, _ = self._process_graph_batch(graph)
        all_atom_feats = []

        for layer in self.layers:
            atom_feats, bond_feats = layer(
                atom_feats, g.edge_index, bond_feats, batch=g.batch
            )
            all_atom_feats.append(atom_feats)

        final_atom_feats = self._run_jk(all_atom_feats)

        return self.readout(g, final_atom_feats)
