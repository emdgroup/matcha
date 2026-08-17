"""Finetuner model for transfer learning from pretrained encoders.

Supports full fine-tuning with layer-wise learning rate decay and
LoRA (Low-Rank Adaptation) as finetuning strategies.
"""

from matcha.torch.models.classic.base_classic_model import ClassicModelRegistry
from matcha.utils.serialization import load_yaml
from matcha.torch.predictors import MLP
from matcha.nn.losses import MultiLoss
from matcha.nn.optimizers import OptimizerRegistry
from matcha.nn.layers import AdaRMSN, GraphNorm
from matcha.torch.models.mixin import ModelMixin
from matcha.torch.models.finetuning.lora import apply_lora
from matcha.torch.models.finetuning.pretrained_encoder_wrapper import (
    PretrainedEncoderWrapper,
)
import os
import torch
from typing import Any
from torch import nn
from lightning.pytorch.core.mixins import HyperparametersMixin
from matcha.utils.schemas import FinetunerInputModel

_SELF_CONTAINED_SENTINEL = "__self_contained__"


def _resolve_leaf_encoder(pretrain: nn.Module) -> nn.Module:
    """Walk through nested Finetuner wrappers to the innermost encoder."""
    while isinstance(pretrain, Finetuner):
        pretrain = pretrain.pretrain
    return pretrain.encoder


@ClassicModelRegistry.register()
class Finetuner(ModelMixin, HyperparametersMixin):
    """Finetuner model that adapts a pretrained encoder for downstream tasks.

    Loads a pretrained model (classic, pretraining, or nested finetuner) and
    attaches a fresh MLP prediction head. Supports two finetuning strategies:

    - ``"full"``: All encoder parameters are trainable with layer-wise learning
      rate decay (deeper layers get smaller LRs). Uses two optimizers — one for
      the new prediction head and one for the pretrained backbone.
    - ``"lora"``: Freezes the pretrained encoder and injects LoRA adapters into
      qualifying linear layers. Uses a single optimizer for the adapters and
      prediction head.

    Normalization layers (BatchNorm, LayerNorm, Embedding, AdaRMSN, GraphNorm)
    are always frozen regardless of strategy.

    :param str architecture: Name of the pretrained model architecture in the
        ClassicModelRegistry (e.g., ``"ginmodel"``, ``"gpsmodel"``)
    :param str path_to_pretrained: Path to the pretrained model artifact directory
    :param list[int] pred_hidden_dims: Hidden dimensions for the new prediction
        head MLP, defaults to [256, 256]
    :param list[int] | None task_head_dims: Per-task head dimensions, defaults to None
    :param str activation: Activation function for prediction head, defaults to 'relu'
    :param float dropout: Dropout rate for prediction head, defaults to 0.1
    :param int num_endpoints: Number of prediction targets, defaults to 1
    :param str loss_fn: Loss function name, defaults to 'mse'
    :param dict loss_args: Additional loss function arguments
    :param str optimizer: Optimizer name, defaults to 'adam'
    :param dict optimizer_args: Optimizer arguments, defaults to {"lr": 1e-4}
    :param float pretrain_lr: Learning rate for pretrained encoder (full strategy),
        defaults to 1e-6
    :param float pretrain_decay: Layer-wise LR decay factor (full strategy),
        defaults to 0.5
    :param str scheduler: Learning rate scheduler name, defaults to 'cosine_annealing'
    :param dict scheduler_args: Scheduler arguments
    :param str finetuning_strategy: Either ``"full"`` or ``"lora"``, defaults to 'full'
    :param int lora_rank: LoRA decomposition rank, defaults to 4
    :param float lora_alpha: LoRA scaling numerator, defaults to 8.0
    :param int lora_min_dim: Minimum layer dimension for LoRA injection, defaults to 32
    :param dict | None _pretrain_config: Internal config for checkpoint reconstruction
        (not a user-facing parameter)
    """

    def __init__(
        self,
        architecture: str,
        path_to_pretrained: str,
        pred_hidden_dims: list[int] = [256, 256],
        task_head_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 1e-4},
        pretrain_lr: float = 1e-6,
        pretrain_decay: float = 0.5,
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {},
        finetuning_strategy: str = "full",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_min_dim: int = 32,
        _pretrain_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._max_task_tracking_n = (
            100  # prevent metric overflow for massive multitask datasets
        )

        # Filter out _pretrain_config before schema validation (not a schema field)
        schema_hparams = {
            k: v for k, v in self.hparams.items() if not k.startswith("_")
        }
        self.params = FinetunerInputModel(**schema_hparams)

        if path_to_pretrained == _SELF_CONTAINED_SENTINEL:
            self._build_skeleton_from_config(_pretrain_config)
        else:
            self._load_from_pretrained_path(path_to_pretrained)

        self.pretrain.predictor.prediction_head = None
        self.automatic_optimization = False
        self._mc_dropout_flag = False
        self._label_names = self.pretrain._label_names

        for module in self.pretrain.modules():
            if isinstance(
                module, (nn.BatchNorm1d, nn.LayerNorm, nn.Embedding, AdaRMSN, GraphNorm)
            ):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        self.pretrain_output_dim = self.pretrain.latent_dim

        self.predictor = MLP(
            input_dim=self.pretrain_output_dim,
            hidden_dims=pred_hidden_dims,
            task_head_dims=task_head_dims,
            num_endpoints=num_endpoints,
            dropout=dropout,
            activation=activation,
            norm="batch",
        )

        self._parse_loss_fn(loss_fn, loss_args, num_endpoints)

        if finetuning_strategy == "lora":
            self._setup_lora(
                optimizer,
                optimizer_args,
                scheduler,
                scheduler_args,
                lora_rank,
                lora_alpha,
                lora_min_dim,
            )
        elif finetuning_strategy == "full":
            self._setup_full(
                architecture,
                optimizer,
                optimizer_args,
                pretrain_lr,
                pretrain_decay,
                scheduler,
                scheduler_args,
            )
        else:
            raise ValueError(
                f"Unknown finetuning_strategy '{finetuning_strategy}'. "
                "Choose from 'full' or 'lora'."
            )

        self._init_metric_containers()
        self._label_names = []

    # ------------------------------------------------------------------
    #  Loading paths
    # ------------------------------------------------------------------

    def _load_from_pretrained_path(self, path_to_pretrained: str) -> None:
        """Load pretrained model from filesystem (training-time path)."""
        self.hparams["path_to_pretrained"] = path_to_pretrained
        self.hparams["pretrain_params"] = load_yaml(
            os.path.join(path_to_pretrained, "config", "model.yaml")
        )

        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        # Detect origin type from manifest
        manifest_path = os.path.join(path_to_pretrained, "config", "manifest.yaml")
        manifest = load_yaml(manifest_path) if os.path.exists(manifest_path) else {}
        origin_type = manifest.get("origin_type", "classic")

        # Store origin metadata for _build_pretrain_config() later
        self.hparams["_origin_type"] = origin_type
        self.hparams["_source_class"] = manifest.get("source_class", None)

        if origin_type == "pretraining":
            from matcha.torch.models.pretraining import PretrainingModelRegistry

            source_class = manifest["source_class"]
            pretrain_params = self.hparams["pretrain_params"].copy()
            pretrain_params.pop("torch_type", None)

            pretrain_model = PretrainingModelRegistry[source_class](**pretrain_params)
            encoder_ckpt_path = os.path.join(path_to_pretrained, "encoder.ckpt")
            encoder_state = torch.load(
                encoder_ckpt_path,
                weights_only=False,
                map_location=torch.device(accelerator),
            )
            pretrain_model.encoder.load_state_dict(encoder_state)

            encoder_type = (
                "mlm"
                if "mlm" in source_class.lower() or "roformer" in source_class.lower()
                else "graph"
            )
            self.pretrain = PretrainedEncoderWrapper(
                pretrain_model.encoder, encoder_type
            )
        elif self.hparams["architecture"] != "finetunermodel":
            architecture = self.hparams["architecture"]
            try:
                self.pretrain = ClassicModelRegistry[architecture].load_from_checkpoint(
                    f"{path_to_pretrained}/model.ckpt"
                )
            except Exception:
                ckpt = torch.load(
                    f"{path_to_pretrained}/model.ckpt",
                    weights_only=False,
                    map_location=torch.device(accelerator),
                )
                state_dict = ckpt.get("state_dict", ckpt)

                model = ClassicModelRegistry[architecture](
                    **self.hparams["pretrain_params"]
                )
                model.load_state_dict(state_dict, strict=False)
                self.pretrain = model
        else:
            self.hparams["_origin_type"] = "finetuner"
            self.pretrain = Finetuner.load_from_checkpoint(
                f"{path_to_pretrained}/model.ckpt"
            )

    def _build_skeleton_from_config(self, config: dict) -> None:
        """Reconstruct module graph from config without filesystem access.

        This is the load-time path: Lightning will overwrite all weights via
        load_state_dict() after __init__ completes, so we only need to build
        modules with correct shapes (weights are random/default-initialized).
        """
        if config is None:
            raise ValueError(
                "Cannot build skeleton: _pretrain_config is None but "
                "path_to_pretrained is '__self_contained__'."
            )

        origin_type = config["origin_type"]
        pretrain_params = config["pretrain_params"]

        # Store in hparams for consistency
        self.hparams["pretrain_params"] = pretrain_params
        self.hparams["_origin_type"] = origin_type
        self.hparams["_source_class"] = config.get("source_class", None)

        if origin_type == "pretraining":
            from matcha.torch.models.pretraining import PretrainingModelRegistry

            source_class = config["source_class"]
            params = pretrain_params.copy()
            params.pop("torch_type", None)

            pretrain_model = PretrainingModelRegistry[source_class](**params)
            encoder_type = (
                "mlm"
                if "mlm" in source_class.lower() or "roformer" in source_class.lower()
                else "graph"
            )
            self.pretrain = PretrainedEncoderWrapper(
                pretrain_model.encoder, encoder_type
            )

        elif origin_type == "classic":
            architecture = self.hparams["architecture"]
            params = pretrain_params.copy()
            params.pop("torch_type", None)
            model = ClassicModelRegistry[architecture](**params)
            self.pretrain = model

        elif origin_type == "finetuner":
            # Recursively build nested finetuner skeleton
            nested_config = config.get("nested_pretrain_config", None)
            nested_hparams = config["nested_hparams"]
            # Reconstruct nested Finetuner with sentinel
            self.pretrain = Finetuner(
                _pretrain_config=nested_config,
                path_to_pretrained=_SELF_CONTAINED_SENTINEL,
                **{
                    k: v
                    for k, v in nested_hparams.items()
                    if k
                    not in (
                        "path_to_pretrained",
                        "_pretrain_config",
                        "pretrain_params",
                        "_origin_type",
                        "_source_class",
                    )
                },
            )
        else:
            raise ValueError(
                f"Unknown origin_type in _pretrain_config: '{origin_type}'"
            )

    def _build_pretrain_config(self) -> dict:
        """Capture metadata needed to reconstruct the module graph at load time.

        This is called at save time to embed the config in the checkpoint.
        """
        origin_type = self.hparams.get("_origin_type", "classic")
        config = {
            "origin_type": origin_type,
            "pretrain_params": self.hparams["pretrain_params"],
            "source_class": self.hparams.get("_source_class", None),
        }

        if origin_type == "finetuner":
            # Recursively capture nested finetuner's config
            nested_finetuner = self.pretrain
            config["nested_pretrain_config"] = nested_finetuner._build_pretrain_config()
            # Store the nested finetuner's hparams needed for reconstruction
            config["nested_hparams"] = {
                k: v
                for k, v in nested_finetuner.hparams.items()
                if k not in ("pretrain_params", "_origin_type", "_source_class")
            }

        return config

    # ------------------------------------------------------------------
    #  Strategy helpers
    # ------------------------------------------------------------------

    def _setup_lora(
        self,
        optimizer: str,
        optimizer_args: dict,
        scheduler: str,
        scheduler_args: dict,
        lora_rank: int,
        lora_alpha: float,
        lora_min_dim: int,
    ):
        """Freeze pretrained model, inject LoRA adapters, single optimizer."""
        # Freeze all pretrained parameters
        for param in self.pretrain.parameters():
            param.requires_grad = False

        # Apply LoRA to encoder linear layers
        lora_params = apply_lora(
            _resolve_leaf_encoder(self.pretrain),
            rank=lora_rank,
            alpha=lora_alpha,
            min_dim=lora_min_dim,
        )
        if not lora_params:
            raise ValueError(
                f"No linear layers found in encoder with both dimensions >= {lora_min_dim}. "
                f"Consider lowering lora_min_dim."
            )

        # Single optimizer: LoRA params + predictor
        self.automatic_optimization = True
        all_trainable = list(self.predictor.parameters()) + lora_params
        self.predictor_optimizer = OptimizerRegistry[optimizer](
            all_trainable, **optimizer_args
        )
        self.pretrain_optimizer = None
        self.predictor_scheduler = self._make_scheduler(
            scheduler, self.predictor_optimizer, scheduler_args
        )
        self.pretrain_scheduler = None

    def _setup_full(
        self,
        architecture: str,
        optimizer: str,
        optimizer_args: dict,
        pretrain_lr: float,
        pretrain_decay: float,
        scheduler: str,
        scheduler_args: dict,
    ):
        """Full fine-tuning with layer-wise learning rate decay."""
        self.automatic_optimization = False
        self.predictor_optimizer = OptimizerRegistry[optimizer](
            self.predictor.parameters(), **optimizer_args
        )

        pretrain_opt_params = self.hparams["pretrain_params"]["optimizer_args"].copy()

        encoder = _resolve_leaf_encoder(self.pretrain)

        # Build parameter groups with different learning rates
        param_groups = []

        # 1. Predictor parameters with pretrain_lr
        if hasattr(self.pretrain, "predictor") and self.pretrain.predictor is not None:
            predictor_params = [
                p for p in self.pretrain.predictor.parameters() if p.requires_grad
            ]
            if predictor_params:
                param_groups.append({"params": predictor_params, "lr": pretrain_lr})

        # 2. Encoder parameters with halving learning rates from bottom to top
        if encoder is not None:
            encoder_layers = list(encoder.layers) if hasattr(encoder, "layers") else []

            if encoder_layers:
                layer_lr = pretrain_lr
                for layer_idx, layer in enumerate(encoder_layers):
                    layer_params = [p for p in layer.parameters() if p.requires_grad]
                    if layer_params:
                        layer_lr = pretrain_lr * (pretrain_decay ** (layer_idx + 1))
                        param_groups.append({"params": layer_params, "lr": layer_lr})

                # Handle any remaining encoder parameters not in layers
                layer_param_ids = {
                    id(p) for layer in encoder_layers for p in layer.parameters()
                }
                other_encoder_params = [
                    p
                    for p in encoder.parameters()
                    if p.requires_grad and id(p) not in layer_param_ids
                ]
                if other_encoder_params:
                    param_groups.append(
                        {"params": other_encoder_params, "lr": layer_lr}
                    )
            else:
                encoder_params = [p for p in encoder.parameters() if p.requires_grad]
                if encoder_params:
                    param_groups.append(
                        {"params": encoder_params, "lr": pretrain_lr * pretrain_decay}
                    )

        # 3. Any remaining pretrain parameters
        all_param_ids = {id(p) for group in param_groups for p in group["params"]}
        remaining_params = [
            p
            for p in self.pretrain.parameters()
            if p.requires_grad and id(p) not in all_param_ids
        ]
        if remaining_params:
            param_groups.append({"params": remaining_params, "lr": pretrain_lr})

        pretrain_opt_params.pop("lr", None)
        self.pretrain_optimizer = OptimizerRegistry[optimizer](
            param_groups, **pretrain_opt_params
        )

        self.predictor_scheduler = self._make_scheduler(
            scheduler, self.predictor_optimizer, scheduler_args
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Simple forward function for the training and prediction step. Depending
        on the architecture, the input can be in different formats.

        :param dict[str, Any] batch: batch of inputs to process

        :return torch.Tensor: batched predictions
        """

        mol_features = self.pretrain.encode(batch)
        return self.predictor.forward(mol_features)

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the learned representation, i.e. the output of the
        hidden layers right before the prediction head."""
        return self.predictor.latent_dim

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Simple encoding function for getting the latent space of a batch. Depending
        on the architecture, the input can be in different formats.

        :param dict[str, Any] batch: batch of inputs to process

        :return torch.Tensor: latent space
        """
        mol_features = self.pretrain.encode(batch)
        return self.predictor.encode(mol_features)

    def compute_learned_embedding(self, x) -> torch.Tensor:
        """Extracts learned embeddings for a given batch from a dataloader.

        :param object x: dataloader object containing the batched dataset to compute
            embeddings of

        :return torch.Tensor: learned embeddings for the dataset
        """
        output = []
        for batch in x:
            mol_features = self.pretrain.encode(batch)
            mol_features = self.predictor.encode(mol_features)
            output.append(mol_features)
        return output

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step logic for finetuning.

        :param dict[str, Any] batch: batch of inputs to process

        :param int batch_idx: leftover from lightning tutorial which I am too scared
            to remove (TODO)

        :return torch.Tensor: batch loss
        """
        y = batch["y"]
        y_pred = self.forward(batch)
        if not isinstance(self.loss_fn, MultiLoss):
            train_loss = self.loss_fn(y_pred, y)
        else:
            train_loss, loss_log = self.loss_fn(y_pred, y, self.global_step)

        if self.pretrain_optimizer is not None:
            # Full fine-tuning: manual optimization with two optimizers.
            # Step through Lightning-wrapped optimizers so
            # `manual_optimization.optim_step_progress` (and hence
            # `self.global_step`) advances — the raw torch refs bypass
            # `LightningOptimizer._on_before_step`/`_on_after_step` hooks.
            opt_predictor, opt_pretrain = self.optimizers()
            self.predictor_optimizer.zero_grad()
            self.pretrain_optimizer.zero_grad()
            self.manual_backward(train_loss)
            opt_predictor.step()
            opt_pretrain.step()

            # Manually step scheduler (Lightning skips auto-stepping
            # when automatic_optimization is False)
            self.predictor_scheduler.step()

        if not isinstance(self.loss_fn, MultiLoss):
            self.log("train_loss", train_loss, prog_bar=True, on_step=True)
        else:
            self.log(
                "train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True
            )
            for name, log in loss_log.items():
                self.log(
                    f"train_{name}_loss",
                    log["loss"],
                    prog_bar=True,
                    on_step=True,
                    sync_dist=True,
                )
                self.log(
                    f"train_{name}_weight",
                    log["weight"],
                    prog_bar=True,
                    on_step=True,
                    sync_dist=True,
                )
        return train_loss

    def predict_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Prediction step logic for classic models.

        :param dict[str, Any] batch: batch of inputs to process

        :return torch.Tensor: predictions for the batch
        """
        # decide whether to keep dropout on or off depending on flag
        if self.mc_dropout_flag:
            for module in self.predictor.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()
        else:
            for module in self.predictor.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
        return self.forward(batch)

    def configure_optimizers(self):
        """Configure optimizers and schedulers for Lightning Trainer.

        Returns two optimizers for full fine-tuning (manual optimization) or
        a single optimizer with scheduler for LoRA (automatic optimization).
        """
        if self.pretrain_optimizer is not None:
            # Full fine-tuning: manual optimization — schedulers are stepped
            # explicitly in training_step, so we only return the optimizers.
            return [self.predictor_optimizer, self.pretrain_optimizer]
        # LoRA: automatic optimization — return optimizer + scheduler so
        # Lightning auto-steps the scheduler each step.
        return {
            "optimizer": self.predictor_optimizer,
            "lr_scheduler": {
                "scheduler": self.predictor_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
