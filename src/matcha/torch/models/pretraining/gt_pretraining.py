"""GT (Graph Transformer) pretraining model for self-supervised learning on graphs."""

from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch
from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.torch.models.pretraining.base_graph_pretraining import (
    BaseGraphPretrainingModel,
)
from matcha.torch.models.pretraining.base_pretraining_model import (
    PretrainingModelRegistry,
)
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from matcha.torch.encoders.gt import GTConv
from matcha.nn.layers import LnBnDr, SpatialEncoder
from matcha.datamodules.classic.graph_datamodule import ATOM_FEAT_DIM, BOND_FEAT_DIM


class GTPretrainingEncoder(BaseGraphEncoder, HyperparametersMixin):
    """GT encoder variant that returns node embeddings for pretraining.

    This encoder outputs node-level embeddings that can be used for both
    node-level and graph-level pretraining tasks.

    :param int num_layers: Number of GTConv layers
    :param int atom_input_dim: Number of input atom features
    :param int bond_input_dim: Number of input bond features
    :param int atom_hidden_dim: Hidden dimension for atom features
    :param int num_heads: Number of attention heads
    :param int expansion_k: FFN expansion factor
    :param int | None distance_k: Maximum distance for spatial encoding
    :param str activation: Activation function name
    :param float dropout: Dropout rate
    :param str jk: Jumping knowledge strategy
    :param str readout: Readout function for graph-level aggregation
    :param int laplacian_k: Laplacian positional encoding dimension
    :param int rwse_k: Random walk structural encoding dimension
    :param int elstatic_k: Electrostatic encoding dimension
    :param int distmat_k: Distance matrix encoding dimension
    :param int rrwp_k: Relative random walk probability dimension
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
        super().__init__(
            laplacian_k,
            rwse_k,
            elstatic_k,
            distmat_k,
            rrwp_k,
        )

        # Snap num_heads to the largest divisor of atom_hidden_dim that is <= num_heads,
        # so HPO can freely sample both values without causing a shape mismatch.
        while num_heads > 1 and atom_hidden_dim % num_heads != 0:
            num_heads -= 1

        self.save_hyperparameters()
        self.num_heads = num_heads
        self.expansion_k = expansion_k

        # Input projections
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

        # Spatial encoder for distance bias
        if distance_k is not None:
            self.dist_encoder = SpatialEncoder(distance_k, num_heads)
        else:
            self.dist_encoder = None

        self._parse_jk(jk)
        self._parse_readout(readout)

    def _get_distance_bias(self, graph: Batch) -> torch.Tensor | None:
        """Compute distance bias for ALL node pairs from the SPD matrix.

        :param graph: Batched PyG graph with spd attribute
        :return: Distance bias or None
        """
        if self.dist_encoder is None or not hasattr(graph, "spd") or graph.spd is None:
            return None
        return self.dist_encoder(graph.spd)

    def forward(self, graph: Batch) -> torch.Tensor:
        """Forward pass returning graph-level embeddings.

        :param graph: Batched PyG graph
        :return: Graph-level embeddings [batch_size, hidden_dim]
        """
        node_embeddings, g = self.forward_nodes(graph)
        return self.readout(g, node_embeddings)

    def forward_nodes(self, graph: Batch) -> tuple[torch.Tensor, Batch]:
        """Forward pass returning node-level embeddings (JK-merged).

        :param graph: Batched PyG graph
        :return: Tuple of (node_embeddings, processed_graph)
        """
        all_atom_feats, g = self.forward_nodes_per_layer(graph)
        final_atom_feats = self._run_jk(all_atom_feats)
        return final_atom_feats, g

    def forward_nodes_per_layer(self, graph: Batch) -> tuple[list[torch.Tensor], Batch]:
        """Forward pass returning per-layer node embeddings.

        :param graph: Batched PyG graph
        :return: Tuple of (per_layer_embeddings, processed_graph)
        """
        g, atom_feats, bond_feats, graph_id = self._process_graph_batch(graph)

        # Project input features to hidden dimension
        atom_feats = self.atom_projection(atom_feats)
        bond_feats = self.bond_projection(bond_feats)

        # Get distance bias for ALL node pairs
        dist_bias = self._get_distance_bias(g)

        all_atom_feats = []

        # Pass through GTConv stack with distance bias
        for layer in self.layers:
            atom_feats = layer(
                atom_feats, g.edge_index, bond_feats, graph_id, dist_bias=dist_bias
            )
            all_atom_feats.append(atom_feats)

        return all_atom_feats, g


@PretrainingModelRegistry.register()
class GTPretraining(BaseGraphPretrainingModel):
    """GT (Graph Transformer) model for self-supervised pretraining.

    Graph Transformer uses multi-head self-attention with edge features and
    shortest-path distance encoding. This model performs a single encoder
    forward pass and produces both:
    - Node-level predictions via a dedicated node MLP head
    - Graph-level predictions via readout + graph MLP head

    Example usage:

    .. code-block:: python

        model = GTPretraining(
            num_node_targets=44,   # e.g., predict atom types
            num_graph_targets=1,   # e.g., predict molecular property
        )

        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param int num_node_targets: Number of node-level prediction targets
    :param int num_graph_targets: Number of graph-level prediction targets
    :param int enc_num_layers: Number of GTConv layers, defaults to 3
    :param int enc_atom_input_dim: Input atom feature dimension, defaults to ATOM_FEAT_DIM
    :param int enc_bond_input_dim: Input bond feature dimension, defaults to BOND_FEAT_DIM
    :param int enc_atom_hidden_dim: Hidden dimension for atoms, defaults to 512
    :param int enc_num_heads: Number of attention heads, defaults to 8
    :param int enc_expansion_k: FFN expansion factor, defaults to 2
    :param int | None enc_distance_k: Maximum distance for spatial encoding
    :param str enc_jk: Jumping knowledge strategy, defaults to 'last'
    :param str enc_readout: Readout function, defaults to 'vpa'
    :param str enc_activation: Activation function, defaults to 'gelu'
    :param float enc_dropout: Dropout rate, defaults to 0.2
    :param list[int] node_head_dims: Hidden dims for shared node prediction head
    :param list[int] graph_head_dims: Hidden dims for shared graph prediction head
    :param list[int] node_task_head_dims: Per-task hidden dims for node head
    :param list[int] graph_task_head_dims: Per-task hidden dims for graph head
    :param str pred_activation: Activation for prediction heads
    :param float pred_dropout: Dropout for prediction heads
    :param str loss_fn: Loss function name
    :param dict loss_args: Loss function arguments
    :param str optimizer: Optimizer name
    :param dict optimizer_args: Optimizer arguments
    :param int | None enc_node_encoder_depth: Number of encoder layers used by the
        node prediction head. When set, the node MLP receives embeddings from
        only the first ``enc_node_encoder_depth`` layers while the graph MLP
        uses all ``enc_num_layers`` layers. Defaults to None (both heads use
        all layers).
    :param str scheduler: Scheduler name
    :param dict scheduler_args: Scheduler arguments
    :param float node_loss_weight: Constant weight for node-level loss
    :param float graph_loss_weight: Constant weight for graph-level loss
    """

    def __init__(
        self,
        num_node_targets: int = 1,
        num_graph_targets: int = 1,
        enc_num_layers: int = 3,
        enc_atom_input_dim: int = ATOM_FEAT_DIM,
        enc_bond_input_dim: int = BOND_FEAT_DIM,
        enc_atom_hidden_dim: int = 512,
        enc_num_heads: int = 8,
        enc_expansion_k: int = 2,
        enc_distance_k: int | None = 5,
        enc_jk: str = "last",
        enc_readout: str = "vpa",
        enc_activation: str = "gelu",
        enc_dropout: float = 0.2,
        enc_laplacian_k: int = 10,
        enc_rwse_k: int = 20,
        enc_elstatic_k: int = 0,
        enc_distmat_k: int = 0,
        enc_rrwp_k: int = 0,
        enc_node_encoder_depth: int | None = None,
        node_head_dims: list[int] | None = None,
        graph_head_dims: list[int] | None = None,
        node_task_head_dims: list[int] | None = None,
        graph_task_head_dims: list[int] | None = None,
        pred_activation: str = "swish",
        pred_dropout: float = 0.2,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adamw",
        optimizer_args: dict = {"lr": 1e-4},
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {"min_lr": 1e-6, "total_steps": 50},
        node_loss_weight: float = 0.5,
        graph_loss_weight: float = 0.5,
        per_task_log_every_n_steps: int = 1,
    ):
        super().__init__(
            num_node_targets=num_node_targets,
            num_graph_targets=num_graph_targets,
            node_head_dims=node_head_dims,
            graph_head_dims=graph_head_dims,
            node_task_head_dims=node_task_head_dims,
            graph_task_head_dims=graph_task_head_dims,
            pred_activation=pred_activation,
            pred_dropout=pred_dropout,
            loss_fn=loss_fn,
            loss_args=loss_args,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            node_loss_weight=node_loss_weight,
            graph_loss_weight=graph_loss_weight,
            per_task_log_every_n_steps=per_task_log_every_n_steps,
        )
        self.save_hyperparameters()

        # Build encoder
        self._build_encoder()

        # Calculate prediction head input dimension
        if enc_jk == "concat":
            head_input_dim = enc_atom_hidden_dim * enc_num_layers
        else:
            head_input_dim = enc_atom_hidden_dim

        # Build node prediction head
        self.node_head = self._build_prediction_head(
            input_dim=head_input_dim,
            hidden_dims=node_head_dims,
            num_targets=num_node_targets,
            dropout=pred_dropout,
            activation=pred_activation,
            task_head_dims=node_task_head_dims,
        )

        # Build graph prediction head
        self.graph_head = self._build_prediction_head(
            input_dim=head_input_dim,
            hidden_dims=graph_head_dims,
            num_targets=num_graph_targets,
            dropout=pred_dropout,
            activation=pred_activation,
            task_head_dims=graph_task_head_dims,
        )

        self._parse_train_config()

    def _build_encoder(self):
        """Build the GT encoder."""
        atom_input_dim = (
            self.hparams["enc_atom_input_dim"]
            + self.hparams["enc_laplacian_k"]
            + self.hparams["enc_rwse_k"]
            + self.hparams["enc_distmat_k"]
            + self.hparams["enc_elstatic_k"]
        )
        edge_input_dim = self.hparams["enc_bond_input_dim"] + self.hparams["enc_rrwp_k"]

        self.encoder = GTPretrainingEncoder(
            num_layers=self.hparams["enc_num_layers"],
            atom_input_dim=atom_input_dim,
            bond_input_dim=edge_input_dim,
            atom_hidden_dim=self.hparams["enc_atom_hidden_dim"],
            num_heads=self.hparams["enc_num_heads"],
            expansion_k=self.hparams["enc_expansion_k"],
            distance_k=self.hparams["enc_distance_k"],
            activation=self.hparams["enc_activation"],
            dropout=self.hparams["enc_dropout"],
            jk=self.hparams["enc_jk"],
            readout=self.hparams["enc_readout"],
            laplacian_k=self.hparams["enc_laplacian_k"],
            rwse_k=self.hparams["enc_rwse_k"],
            elstatic_k=self.hparams["enc_elstatic_k"],
            distmat_k=self.hparams["enc_distmat_k"],
            rrwp_k=self.hparams["enc_rrwp_k"],
        )

    def _get_per_layer_embeddings(
        self, batch: dict[str, Any]
    ) -> tuple[list[torch.Tensor], Batch]:
        """Extract per-layer node embeddings from GT encoder.

        :param batch: Input batch containing 'graph' key
        :return: Tuple of (per_layer_embeddings, processed_graph)
        """
        graph = batch["graph"]
        return self.encoder.forward_nodes_per_layer(graph)
