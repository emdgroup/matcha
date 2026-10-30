"""GPS pretraining model for self-supervised learning on graphs."""

from matcha.torch.encoders.gps import GPS
from matcha.torch.models.pretraining.base_graph_pretraining import (
    BaseGraphPretrainingModel,
)
from matcha.torch.models.pretraining.base_pretraining_model import (
    PretrainingModelRegistry,
)
from matcha.datamodules.classic.graph_datamodule import ATOM_FEAT_DIM, BOND_FEAT_DIM


@PretrainingModelRegistry.register()
class GPSPretraining(BaseGraphPretrainingModel):
    """GPS model for self-supervised pretraining with node and graph level predictions.

    GPS (General, Powerful, Scalable) combines local message passing with
    global self-attention. This model performs a single encoder forward pass
    and produces both:
    - Node-level predictions via a dedicated node MLP head
    - Graph-level predictions via readout + graph MLP head

    Example usage:

    .. code-block:: python

        model = GPSPretraining(
            num_node_targets=44,   # e.g., predict atom types
            num_graph_targets=1,   # e.g., predict molecular property
        )

        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param int num_node_targets: Number of node-level prediction targets
    :param int num_graph_targets: Number of graph-level prediction targets
    :param int enc_num_layers: Number of GPS blocks, defaults to 6
    :param int enc_atom_input_dim: Input atom feature dimension, defaults to ATOM_FEAT_DIM
    :param int enc_bond_input_dim: Input bond feature dimension, defaults to BOND_FEAT_DIM
    :param int enc_atom_hidden_dim: Hidden dimension for atoms, defaults to 512
    :param int enc_num_heads: Number of attention heads, defaults to 8
    :param int enc_expansion_k: FFN expansion factor, defaults to 1
    :param int | None enc_distance_k: Maximum distance for spatial encoding
    :param str enc_jk: Jumping knowledge strategy, defaults to 'last'
    :param str | None enc_norm: Normalization type, defaults to 'layer'
    :param str enc_readout: Readout function, defaults to 'vpa'
    :param str enc_activation: Activation function, defaults to 'swish'
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
        enc_num_layers: int = 6,
        enc_atom_input_dim: int = ATOM_FEAT_DIM,
        enc_bond_input_dim: int = BOND_FEAT_DIM,
        enc_atom_hidden_dim: int = 512,
        enc_num_heads: int = 8,
        enc_expansion_k: int = 1,
        enc_distance_k: int | None = None,
        enc_jk: str = "last",
        enc_norm: str | None = "layer",
        enc_readout: str = "vpa",
        enc_activation: str = "swish",
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
        """Build the canonical GPS encoder shared with the classic path."""
        atom_input_dim = (
            self.hparams["enc_atom_input_dim"]
            + self.hparams["enc_laplacian_k"]
            + self.hparams["enc_rwse_k"]
            + self.hparams["enc_distmat_k"]
            + self.hparams["enc_elstatic_k"]
        )
        edge_input_dim = self.hparams["enc_bond_input_dim"] + self.hparams["enc_rrwp_k"]

        self.encoder = GPS(
            num_layers=self.hparams["enc_num_layers"],
            atom_input_dim=atom_input_dim,
            bond_input_dim=edge_input_dim,
            atom_hidden_dim=self.hparams["enc_atom_hidden_dim"],
            num_heads=self.hparams["enc_num_heads"],
            expansion_k=self.hparams["enc_expansion_k"],
            distance_k=self.hparams["enc_distance_k"],
            activation=self.hparams["enc_activation"],
            dropout=self.hparams["enc_dropout"],
            norm=self.hparams["enc_norm"],
            jk=self.hparams["enc_jk"],
            readout=self.hparams["enc_readout"],
            laplacian_k=self.hparams["enc_laplacian_k"],
            rwse_k=self.hparams["enc_rwse_k"],
            elstatic_k=self.hparams["enc_elstatic_k"],
            distmat_k=self.hparams["enc_distmat_k"],
            rrwp_k=self.hparams["enc_rrwp_k"],
        )
