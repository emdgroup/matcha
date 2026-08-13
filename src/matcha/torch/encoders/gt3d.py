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
from matcha.utils.logging import get_default_logger

_LOGGER = get_default_logger(__name__)


@EncoderRegistry.register()
class GT3D(BaseGraphEncoder, HyperparametersMixin):
    """3D-aware Graph Transformer encoder using stacked GTConv layers with 3D distance-biased attention.

    A graph transformer architecture that uses multi-head self-attention
    with edge features and 3D Gaussian Basis Kernel spatial encoding as attention bias.

    This encoder extends GT by using 3D coordinate-based spatial encoding
    instead of graph-based shortest path distances. Uses only the 3D pathway;
    no 2D shortest-path-distance bias is applied.

    Architecture: atom_projection, bond_projection → GTConv stack (with 3D distance bias) → readout

    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.

    References:
    - https://arxiv.org/abs/2210.01765 §3.2 (Uni-Mol: Gaussian Basis Kernel spatial encoding)

    It is intended to be used inside a :class:`BaseClassicModel` instance.

    :param int num_layers: number of GTConv layers
    :param int atom_input_dim: number of input atom features after positional
        encoding concatenation (used by the atom projection)
    :param int raw_atom_input_dim: number of raw input atom features from the
        featurizer, before positional encoding concatenation (used by the 3D
        spatial encoder to compute γ/β from the pre-PE features)
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
        raw_atom_input_dim: int,
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
        requested_num_heads = num_heads
        while num_heads > 1 and atom_hidden_dim % num_heads != 0:
            num_heads -= 1
        if num_heads != requested_num_heads:
            _LOGGER.warning(
                "GT3D: num_heads=%d does not divide atom_hidden_dim=%d; "
                "snapping to num_heads=%d.",
                requested_num_heads,
                atom_hidden_dim,
                num_heads,
            )
        self.save_hyperparameters()

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

        # 3D Spatial Encoder using Gaussian Basis Kernels.
        # γ/β are computed from the raw pre-PE atom features (per Uni-Mol §3.2),
        # not from the projected hidden features.
        self.dist3d_encoder = SpatialEncoder3d(
            num_kernels, num_heads, atom_feat_dim=raw_atom_input_dim
        )

        self._parse_jk(jk)
        self._parse_readout(readout)

    def forward_nodes_per_layer(
        self, graph: Batch
    ) -> tuple[list[torch.Tensor], Batch]:
        """Run 3D-biased graph-transformer message passing and return one
        node-feature tensor per layer.

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
                "GT3D requires 3D coordinates on graph.pos, but the batch "
                "has none. Attach per-node coordinates to Data.pos in the "
                "datamodule (see Graph3DDataModule) before calling the encoder."
            )
        # Capture the raw pre-PE, pre-projection atom features. `_process_graph_batch`
        # clones `batch.x` before concatenating positional encodings onto its return
        # value, so `graph.x` itself remains the raw featurizer output.
        raw_atom_feats = graph.x
        g, atom_feats, bond_feats, graph_id = self._process_graph_batch(graph)

        # Project input features to hidden dimension
        atom_feats = self.atom_projection(atom_feats)
        bond_feats = self.bond_projection(bond_feats)

        # Convert coordinates and raw atom features to dense batch format for 3D
        # spatial encoding. coords: [num_atoms, 3] -> [batch_size, max_nodes, 3];
        # raw_atom_feats: [num_atoms, raw_atom_input_dim] -> [batch_size, max_nodes, raw_atom_input_dim].
        coords_dense, _ = to_dense_batch(coords, graph_id)
        raw_atom_feats_dense, _ = to_dense_batch(raw_atom_feats, graph_id)

        # Compute 3D spatial encoding bias from coords + raw pre-PE features.
        # Output shape: [batch_size, max_nodes, max_nodes, num_heads]
        dist_bias = self.dist3d_encoder(coords_dense, raw_atom_feats_dense)

        all_atom_feats = []

        # Pass through GTConv stack with 3D distance bias
        for layer in self.layers:
            atom_feats = layer(
                atom_feats, g.edge_index, bond_feats, graph_id, dist_bias=dist_bias
            )
            all_atom_feats.append(atom_feats)

        return all_atom_feats, g
