"""Base class for graph neural network pretraining models.

Provides a framework for GNN models that jointly predict user-provided
atom-level and molecule-level labels through a shared encoder with separate
prediction heads.
"""

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from lightning.pytorch.core.mixins import HyperparametersMixin

from matcha.torch.models.pretraining.base_pretraining_model import (
    BasePretrainingModel,
)
from matcha.nn.losses import MultitaskLoss, GradNormLoss, MultiLoss
from matcha.nn.layers import LnBnDr, MultiMLP


class BaseGraphPretrainingModel(BasePretrainingModel, HyperparametersMixin):
    """Base class for graph neural network pretraining models.

    This class provides a framework for GNN models that support simultaneous
    node-level and graph-level predictions. A single forward pass through the
    encoder produces node embeddings, which are then:
    - Passed to a node-level MLP for node predictions
    - Aggregated via readout and passed to a graph-level MLP for graph predictions

    Both node-level and graph-level targets are externally provided by the user
    (e.g. computed atom properties like partial charges, SASA, or electronegativity
    for nodes, and molecular descriptors like logP or molecular weight for graphs).
    The model does **not** reconstruct its own input atom features.

    Subclasses must implement:
        - _build_encoder(): Build the graph encoder. The encoder must be a
          :class:`BaseGraphEncoder` subclass; per-layer node embeddings are
          then read from ``self.encoder.forward_nodes_per_layer`` — pretraining
          models do not implement their own layer stack.

    :param int num_node_targets: Number of node-level prediction targets
    :param int num_graph_targets: Number of graph-level prediction targets
    :param list[int] node_head_dims: Hidden dimensions for the shared node prediction
        head MLP (applied to all node tasks jointly)
    :param list[int] graph_head_dims: Hidden dimensions for the shared graph prediction
        head MLP (applied to all graph tasks jointly)
    :param list[int] node_task_head_dims: Hidden dimensions for per-task node MLPs.
        When set, each node task gets its own independent MLP on top of the shared
        node head layers. Uses ``MultiMLP`` with separate parameters per task.
    :param list[int] graph_task_head_dims: Hidden dimensions for per-task graph MLPs.
        When set, each graph task gets its own independent MLP on top of the shared
        graph head layers. Uses ``MultiMLP`` with separate parameters per task.
    :param str pred_activation: Activation function for prediction heads
    :param float pred_dropout: Dropout rate for prediction heads
    :param str loss_fn: Loss function name (used for both heads)
    :param dict loss_args: Additional loss function arguments
    :param str optimizer: Optimizer name
    :param dict optimizer_args: Optimizer arguments
    :param str scheduler: Learning rate scheduler name
    :param dict scheduler_args: Scheduler arguments
    :param float node_loss_weight: Constant weight for node-level loss
    :param float graph_loss_weight: Constant weight for graph-level loss
    :param int per_task_log_every_n_steps: How often to log per-task losses.
        Set higher to reduce overhead with many tasks (e.g. >100).
    """

    def __init__(
        self,
        num_node_targets: int = 1,
        num_graph_targets: int = 1,
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
        super().__init__()
        self._num_node_targets = num_node_targets
        self._num_graph_targets = num_graph_targets
        self._per_task_log_every_n_steps = per_task_log_every_n_steps
        self._per_task_log_step_counter = 0

    @property
    def num_node_targets(self) -> int:
        """Number of node-level prediction targets."""
        return self._num_node_targets

    @property
    def num_graph_targets(self) -> int:
        """Number of graph-level prediction targets."""
        return self._num_graph_targets

    # ------------------------------------------------------------------
    # Loss parsing — override base to wrap in MultitaskLoss for NaN safety
    # ------------------------------------------------------------------

    def _parse_train_config(self):
        """Parse training config, wrapping the loss in MultitaskLoss.

        Graph pretraining targets (both node-level and graph-level) may
        contain NaN entries.  ``MultitaskLoss`` automatically masks those
        out, matching the behaviour used by the classical models.
        """
        super()._parse_train_config()

        # Re-wrap the loss produced by the base class so NaN targets are
        # handled correctly.  The base class creates a raw loss (e.g.
        # nn.MSELoss); we replace it with the MultitaskLoss wrapper that
        # applies NaN masking, unless the user already requested an
        # explicitly NaN-aware loss variant.
        loss_fn = self.hparams["loss_fn"]
        loss_args = self.hparams.get("loss_args", {})

        if loss_fn == "multiloss":
            self.loss_fn = MultiLoss(loss_args["loss_configs"])
        elif loss_fn == "gradnorm":
            inner_fn = loss_args.get("loss_fn", "mse")
            inner_args = loss_args.get("loss_args", {})
            num_ep = loss_args.get("num_endpoints", 1)
            self.loss_fn = GradNormLoss(
                loss_fn=inner_fn,
                loss_args=inner_args,
                num_endpoints=num_ep,
            )
        elif loss_fn == "multitask":
            inner_fn = loss_args.get("loss_fn", "mse")
            inner_args = loss_args.get("loss_args", {})
            self.loss_fn = MultitaskLoss(loss_fn=inner_fn, loss_args=inner_args)
        else:
            # Default: wrap in MultitaskLoss for NaN-safe multi-target handling
            self.loss_fn = MultitaskLoss(loss_fn=loss_fn, loss_args=loss_args)

        self._build_node_loss()

    def _build_node_loss(self) -> None:
        """Build a NaN-safe MultitaskLoss for flat atom-level averaging."""
        loss_fn = self.hparams["loss_fn"]
        loss_args = self.hparams.get("loss_args", {})

        if loss_fn in ("multiloss", "gradnorm", "multitask"):
            inner_fn = loss_args.get("loss_fn", "mse")
            inner_args = loss_args.get("loss_args", {})
        else:
            inner_fn = loss_fn
            inner_args = loss_args

        self.node_loss_fn = MultitaskLoss(loss_fn=inner_fn, loss_args=inner_args)

    @abstractmethod
    def _build_encoder(self):
        """Build the graph encoder. Must set self.encoder."""
        pass

    def _get_per_layer_embeddings(
        self, batch: dict[str, Any]
    ) -> tuple[list[torch.Tensor], Batch]:
        """Extract per-layer node embeddings from the encoder.

        Delegates to :meth:`BaseGraphEncoder.forward_nodes_per_layer` so the
        pretraining path consumes the canonical encoder's layer stack rather
        than re-implementing it.

        :param batch: Input batch containing graph data
        :return: Tuple of (per_layer_embeddings, processed_graph)
            - per_layer_embeddings: List of [num_nodes, hidden_dim] tensors,
              one per encoder layer
            - processed_graph: PyG Batch object for accessing batch assignment
        """
        return self.encoder.forward_nodes_per_layer(batch["graph"])

    def _build_prediction_head(
        self,
        input_dim: int,
        hidden_dims: list[int] | None,
        num_targets: int,
        dropout: float,
        activation: str,
        task_head_dims: list[int] | None = None,
    ) -> nn.Module:
        """Build a prediction head MLP, optionally with per-task branches.

        When ``task_head_dims`` is ``None``, builds a single shared MLP that
        outputs all targets jointly (original behaviour). When
        ``task_head_dims`` is provided, the shared ``hidden_dims`` layers are
        followed by a :class:`~matcha.nn.layers.MultiMLP` with independent
        parameters per target.

        :param input_dim: Input dimension from encoder
        :param hidden_dims: Shared hidden layer dimensions (applied before
            per-task branches). Can be ``None`` to skip shared layers.
        :param num_targets: Number of output targets
        :param dropout: Dropout rate
        :param activation: Activation function name
        :param task_head_dims: Per-task hidden layer dimensions. When set,
            each target gets its own MLP branch on top of the shared layers.
        :return: MLP module
        """
        if hidden_dims is not None:
            layers: list[nn.Module] = []
            dim_list = [input_dim] + hidden_dims
            for i in range(len(dim_list) - 1):
                layers.append(
                    LnBnDr(dim_list[i], dim_list[i + 1], dropout, activation, "batch")
                )
            shared_out_dim = dim_list[-1]
        else:
            layers = []
            shared_out_dim = input_dim

        if task_head_dims is not None:
            task_head = MultiMLP(
                input_dim=shared_out_dim,
                dims=task_head_dims,
                num_parallel=num_targets,
                dropout=dropout,
                activation=activation,
                norm="multibatch",
            )
            layers.append(task_head)
        else:
            layers.append(nn.Linear(shared_out_dim, num_targets))

        return nn.Sequential(*layers)

    def _apply_jk_to_layers(self, layer_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Apply the encoder's jumping knowledge strategy to a list of layer outputs.

        :param layer_embeddings: List of per-layer node embeddings
        :return: Merged node embeddings
        """
        return self.encoder._run_jk(layer_embeddings)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass for pretraining.

        Runs a single encoder forward pass and computes both node-level
        and graph-level predictions. When ``enc_node_encoder_depth`` is set,
        the node head receives embeddings from only the first K layers while
        the graph head receives embeddings from all layers.

        :param batch: Input batch containing graph data
        :return: Dictionary with 'node' and 'graph' predictions
            - 'node': [num_nodes, num_node_targets]
            - 'graph': [batch_size, num_graph_targets]
        """
        # Single encoder forward pass — returns per-layer embeddings
        all_layer_embeddings, graph = self._get_per_layer_embeddings(batch)

        # Node branch: use first node_encoder_depth layers (or all if None)
        node_depth = self.hparams.get("enc_node_encoder_depth")
        if node_depth is not None:
            node_layer_embeddings = all_layer_embeddings[:node_depth]
        else:
            node_layer_embeddings = all_layer_embeddings

        node_embeddings = self._apply_jk_to_layers(node_layer_embeddings)
        node_predictions = self.node_head(node_embeddings)

        # Graph branch: use all layers
        graph_node_embeddings = self._apply_jk_to_layers(all_layer_embeddings)
        graph_embeddings = self.encoder.readout(graph, graph_node_embeddings)
        graph_predictions = self.graph_head(graph_embeddings)

        return {
            "node": node_predictions,
            "graph": graph_predictions,
        }

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract learned representations.

        :param batch: Input batch containing graph data
        :return: Dictionary with 'node' and 'graph' embeddings
            - 'node': [num_nodes, hidden_dim]
            - 'graph': [batch_size, hidden_dim]
        """
        all_layer_embeddings, graph = self._get_per_layer_embeddings(batch)

        node_depth = self.hparams.get("enc_node_encoder_depth")
        if node_depth is not None:
            node_layer_embeddings = all_layer_embeddings[:node_depth]
        else:
            node_layer_embeddings = all_layer_embeddings

        node_embeddings = self._apply_jk_to_layers(node_layer_embeddings)
        graph_node_embeddings = self._apply_jk_to_layers(all_layer_embeddings)
        graph_embeddings = self.encoder.readout(graph, graph_node_embeddings)

        return {
            "node": node_embeddings,
            "graph": graph_embeddings,
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for graph pretraining.

        Expects batch to contain:
            - Graph data (depends on encoder)
            - 'y_node': Node-level targets [num_nodes, num_node_targets]
              (externally provided atom-level labels, e.g. partial charges)
            - 'y_graph': Graph-level targets [batch_size, num_graph_targets]
              (externally provided molecule-level labels, e.g. logP)

        :param batch: Input batch
        :param batch_idx: Batch index
        :return: Total loss value
        """
        predictions = self.forward(batch)

        node_loss = self.node_loss_fn(predictions["node"], batch["y_node"])
        graph_loss = self.loss_fn(predictions["graph"], batch["y_graph"])

        total_loss = (
            self.hparams.node_loss_weight * node_loss
            + self.hparams.graph_loss_weight * graph_loss
        )

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log(
            "train_node_loss", node_loss, prog_bar=False, on_step=True, sync_dist=True
        )
        self.log(
            "train_graph_loss", graph_loss, prog_bar=False, on_step=True, sync_dist=True
        )

        self._log_per_task_losses("train", self.node_loss_fn, self.loss_fn)

        return total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step for graph pretraining.

        :param batch: Input batch
        :param batch_idx: Batch index
        :return: Total loss value
        """
        predictions = self.forward(batch)

        node_loss = self.node_loss_fn(predictions["node"], batch["y_node"])
        graph_loss = self.loss_fn(predictions["graph"], batch["y_graph"])

        # Unweighted validation loss
        total_loss = node_loss + graph_loss

        # Weighted validation loss (matches training weighting)
        weighted_loss = (
            self.hparams.node_loss_weight * node_loss
            + self.hparams.graph_loss_weight * graph_loss
        )

        self.log("val_loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_loss_weighted", weighted_loss, on_epoch=True, sync_dist=True)
        self.log(
            "val_node_loss", node_loss, prog_bar=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "val_graph_loss", graph_loss, prog_bar=False, on_epoch=True, sync_dist=True
        )

        self._log_per_task_losses("val", self.node_loss_fn, self.loss_fn)

        return total_loss

    def _log_per_task_losses(
        self, phase: str, node_loss_fn: torch.nn.Module, graph_loss_fn: torch.nn.Module
    ) -> None:
        """Log individual per-task losses from loss functions that expose them.

        Logging frequency is controlled by ``per_task_log_every_n_steps``. The
        step counter is shared across training and validation phases.

        :param str phase: Logging phase prefix, e.g. ``"train"`` or ``"val"``.
        :param torch.nn.Module node_loss_fn: Node-level loss function.
        :param torch.nn.Module graph_loss_fn: Graph-level loss function.
        """
        self._per_task_log_step_counter += 1
        if self._per_task_log_step_counter % self._per_task_log_every_n_steps != 0:
            return

        on_step = phase == "train"
        on_epoch = phase == "val"
        metrics: dict[str, float] = {}

        for loss_type, loss_fn in [
            ("node_loss", node_loss_fn),
            ("graph_loss", graph_loss_fn),
        ]:
            if hasattr(loss_fn, "_per_task_losses"):
                per_task = loss_fn._per_task_losses.tolist()
                for i, val in enumerate(per_task):
                    metrics[f"{phase}_{loss_type}_col_{i}"] = val

        if metrics:
            self.log_dict(
                metrics,
                prog_bar=False,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
            )
