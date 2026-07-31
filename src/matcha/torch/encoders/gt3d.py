"""3D-aware Graph Transformer encoder using Gaussian Basis Kernel spatial encoding."""

import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from matcha.torch.encoders.gt import GTConv
from matcha.nn.layers import (
    LnBnDr,
    SpatialEncoder3d,
)


@EncoderRegistry.register()
class GT3D(BaseGraphEncoder, HyperparametersMixin):
    """3D-aware Graph Transformer encoder using stacked GTConv layers with 3D distance-biased attention.

    A graph transformer architecture that uses multi-head self-attention
    with edge features and 3D Gaussian Basis Kernel spatial encoding as attention bias.

    This encoder extends GT by using 3D coordinate-based spatial encoding
    instead of graph-based shortest path distances.

    Architecture: atom_projection, bond_projection → GTConv stack (with 3D distance bias) → readout

    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.

    References:
    - https://arxiv.org/abs/2210.01765 (Uni-Mol: One Transformer Can Understand Both 2D & 3D Molecular Data)
    - https://arxiv.org/abs/2012.09699 (Graph Transformer)
    - https://arxiv.org/abs/2205.12454 (GPS)

    It is intended to be used inside a :class:`BaseClassicModel` instance.

    :param int num_layers: number of GTConv layers
    :param int atom_input_dim: number of input atom features from GraphFeaturizer
    :param int bond_input_dim: number of input bond features from GraphFeaturizer
    :param int atom_hidden_dim: hidden dimension for atom features in GTConv layers
    :param int num_heads: number of attention heads, must divide atom_hidden_dim evenly
    :param int expansion_k: expansion factor for the feed-forward network in GTConv layers
    :param int num_kernels: number of Gaussian Basis Kernels for 3D spatial encoding
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
        num_kernels: int,
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
        while num_heads > 1 and atom_hidden_dim % num_heads != 0:
            num_heads -= 1
        self.save_hyperparameters()
        self.num_heads = num_heads
        self.expansion_k = expansion_k

        # Input projections (similar to GT)
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

        # 3D Spatial Encoder using Gaussian Basis Kernels
        self.dist3d_encoder = SpatialEncoder3d(
            num_kernels, num_heads, atom_feat_dim=atom_hidden_dim
        )

        self._parse_jk(jk)
        self._parse_readout(readout)

    def forward(self, graph: Batch, coords: torch.Tensor) -> torch.Tensor:
        """Converts a batched PyG graph and 3D coordinates into a (batch_size, fp_dim) tensor.

        :param Batch graph: batched PyG graph from the dataloader
        :param torch.Tensor coords: batched 3D coordinates [num_atoms, 3] from the dataloader
        :return torch.Tensor: learned molecular representation
        """
        g, atom_feats, bond_feats, graph_id = self._process_graph_batch(graph)

        # Project input features to hidden dimension
        atom_feats = self.atom_projection(atom_feats)
        bond_feats = self.bond_projection(bond_feats)

        # Convert coordinates and atom features to dense batch format for 3D spatial encoding
        # coords: [num_atoms, 3] -> coords_dense: [batch_size, max_nodes, 3]
        coords_dense, _ = to_dense_batch(coords, graph_id)
        atom_feats_dense, _ = to_dense_batch(atom_feats, graph_id)

        # Compute 3D spatial encoding bias
        # Output shape: [batch_size, max_nodes, max_nodes, num_heads]
        dist_bias = self.dist3d_encoder(coords_dense, atom_feats_dense)

        all_atom_feats = []

        # Pass through GTConv stack with 3D distance bias
        for layer in self.layers:
            atom_feats = layer(
                atom_feats, g.edge_index, bond_feats, graph_id, dist_bias=dist_bias
            )
            all_atom_feats.append(atom_feats)

        # Apply jumping knowledge
        final_atom_feats = self._run_jk(all_atom_feats)

        return self.readout(g, final_atom_feats)
