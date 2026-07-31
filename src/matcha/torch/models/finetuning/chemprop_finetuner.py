"""Chemprop-specific finetuner for transfer learning from pretrained D-MPNN models."""

import warnings

from matcha.utils.serialization import load_yaml
from matcha.torch.models.classic import ChempropModel
from matcha.torch.models.finetuning.finetuner import _SELF_CONTAINED_SENTINEL
import os
import torch
from torch import nn
from chemprop.nn.metrics import LossFunctionRegistry as ChempropLossRegistry
from matcha.utils.schemas import ChempropFinetunerInputModel
from chemprop.nn.predictors import (
    RegressionFFN,
    BinaryClassificationFFN,
)
from chemprop.nn import BondMessagePassing
from matcha.torch.models.classic.base_classic_model import ClassicModelRegistry


@ClassicModelRegistry.register()
class ChempropFinetuner(ChempropModel):
    """Finetuner for Chemprop D-MPNN pretrained models.

    Loads a pretrained :class:`ChempropModel` checkpoint and configures the
    prediction head (FFN) based on the loaded predictor state and the requested
    ``pred_hidden_dim``. Normalization layers are frozen after loading. Supports
    loading both full-model checkpoints and raw ``BondMessagePassing``
    (encoder-only) checkpoints.

    Predictor configuration follows four cases:

    - **No predictor loaded, pred_hidden_dim is None**: Creates a linear head
      (single linear layer from encoder output to ``num_endpoints``) using the
      appropriate chemprop FFN class with ``n_layers=0``.
    - **No predictor loaded, pred_hidden_dim is set**: Creates a new FFN with the
      specified hidden dimensions and number of layers.
    - **Predictor loaded, pred_hidden_dim is None**: Keeps the pretrained FFN
      weights but resizes the output layer to ``num_endpoints``. Handles FFN
      type mismatch (regression vs classification) by replacing with the correct
      type and transferring compatible weights.
    - **Predictor loaded, pred_hidden_dim is set**: Replaces the predictor
      entirely with a new FFN of the specified dimensions.

    :param str path_to_pretrained: Path to the pretrained Chemprop artifact directory
    :param int | None pred_hidden_dim: Hidden dimension for the FFN. When None,
        creates a linear head (encoder-only) or keeps pretrained weights (full
        checkpoint). Defaults to 300.
    :param int pred_num_layers: Number of FFN layers, defaults to 2
    :param float pred_dropout: Dropout rate for the FFN, defaults to 0.2
    :param str pred_activation: Activation function for the FFN, defaults to 'relu'
    :param int num_endpoints: Number of prediction targets, defaults to 1
    :param str loss_fn: Loss function name, defaults to 'mse'
    :param str optimizer: Optimizer name, defaults to 'chemprop'
    :param dict optimizer_args: Optimizer arguments, defaults to {"lr": 1e-5}
    :param dict scheduler_args: Scheduler arguments with warmup configuration
    """

    def __init__(
        self,
        path_to_pretrained: str,
        pred_hidden_dim: int | None = 300,
        pred_num_layers: int = 2,
        pred_dropout: float = 0.2,
        pred_activation: str = "relu",
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        optimizer: str = "chemprop",
        optimizer_args: dict = {"lr": 1e-5},
        scheduler_args: dict = {"warmup_epochs": 5, "max_lr": 1e-4, "final_lr": 1e-5},
        _pretrain_config: dict | None = None,
    ):
        # Resolve pretrained model parameters — either from filesystem or config
        if path_to_pretrained == _SELF_CONTAINED_SENTINEL:
            if _pretrain_config is None:
                raise ValueError(
                    "Cannot build skeleton: _pretrain_config is None but "
                    "path_to_pretrained is '__self_contained__'."
                )
            pretrained_params = _pretrain_config["pretrain_params"]
        else:
            pretrained_params = load_yaml(
                os.path.join(path_to_pretrained, "config", "model.yaml")
            )
            pretrained_params.pop("torch_type")

        super().__init__(**pretrained_params)
        self.params = ChempropFinetunerInputModel(
            path_to_pretrained=path_to_pretrained,
            pred_hidden_dim=pred_hidden_dim,
            pred_num_layers=pred_num_layers,
            pred_dropout=pred_dropout,
            pred_activation=pred_activation,
            num_endpoints=num_endpoints,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler_args=scheduler_args,
        )
        self.hparams["pretrain_params"] = pretrained_params
        self.hparams["path_to_pretrained"] = path_to_pretrained

        # Load pretrained weights (skipped for self-contained — Lightning
        # restores weights via load_state_dict after __init__)
        if path_to_pretrained != _SELF_CONTAINED_SENTINEL:
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                ckpt = torch.load(
                    f"{path_to_pretrained}/model.ckpt",
                    weights_only=False,
                    map_location=torch.device(accelerator),
                )
                state_dict = ckpt.get("state_dict", ckpt)
                # strict=False is safe here: the model was constructed with
                # pretrained_params so all shapes match. The predictor may be
                # rebuilt later (Cases C/D), but at this point it still has
                # the original architecture and dimensions.
                self.load_state_dict(state_dict, strict=False)

                for module in self.modules():
                    if isinstance(module, (nn.BatchNorm1d)) or isinstance(
                        module, (nn.LayerNorm)
                    ):
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False

            except Exception:
                ckpt = torch.load(
                    f"{path_to_pretrained}/model.ckpt",
                    weights_only=True,
                    map_location=torch.device(accelerator),
                )
                mp = BondMessagePassing(**ckpt["hyper_parameters"])
                mp.load_state_dict(ckpt["state_dict"])
                self.message_passing = mp
                self.predictor = None

        self.init_lr = optimizer_args["lr"]
        self.max_lr = scheduler_args["max_lr"]
        self.final_lr = scheduler_args["final_lr"]
        self.warmup_epochs = scheduler_args["warmup_epochs"]

        # Determine the correct FFN class for the target loss function
        if loss_fn in ("bce", "ce"):
            TargetFFN = BinaryClassificationFFN
        else:
            TargetFFN = RegressionFFN

        if self.predictor is None and pred_hidden_dim is None:
            # Case A: Encoder-only checkpoint, no hidden dims specified
            # → create a linear head (single linear layer: encoder_dim → num_endpoints)
            self.predictor = TargetFFN(
                input_dim=self.message_passing.output_dim,
                hidden_dim=num_endpoints,
                n_layers=0,
                dropout=0.0,
                activation=pred_activation,
                n_tasks=num_endpoints,
                criterion=ChempropLossRegistry[loss_fn](),
            )
            self.metrics = nn.ModuleList(
                [self.predictor._T_default_metric(), self.criterion.clone()]
            )

        elif self.predictor is None and pred_hidden_dim is not None:
            # Case B: Encoder-only checkpoint, hidden dims specified
            # → create a new FFN with specified architecture
            self.predictor = TargetFFN(
                input_dim=self.message_passing.output_dim,
                hidden_dim=pred_hidden_dim,
                n_layers=pred_num_layers,
                dropout=pred_dropout,
                activation=pred_activation,
                n_tasks=num_endpoints,
                criterion=ChempropLossRegistry[loss_fn](),
            )
            self.metrics = nn.ModuleList(
                [self.predictor._T_default_metric(), self.criterion.clone()]
            )

        elif self.predictor is not None and pred_hidden_dim is None:
            # Case C: Full checkpoint, no hidden dims specified
            # → keep pretrained FFN, resize output layer to num_endpoints
            if isinstance(self.predictor, TargetFFN):
                # Same FFN type — just swap criterion and resize output layer
                self.predictor.criterion = ChempropLossRegistry[loss_fn]()
                new_layer = nn.Linear(
                    pretrained_params["pred_hidden_dim"], num_endpoints
                )
                self.predictor.ffn[-1][-1] = new_layer
            else:
                # FFN type mismatch — instantiate correct type and transfer weights
                warnings.warn(
                    f"Pretrained predictor is {type(self.predictor).__name__} but "
                    f"loss_fn='{loss_fn}' requires {TargetFFN.__name__}. "
                    f"Replacing predictor with correct FFN type and transferring "
                    f"compatible weights.",
                    stacklevel=2,
                )
                old_state_dict = self.predictor.ffn.state_dict()
                new_predictor = TargetFFN(
                    input_dim=self.predictor.ffn.input_dim,
                    hidden_dim=pretrained_params["pred_hidden_dim"],
                    n_layers=pretrained_params["pred_num_layers"],
                    dropout=pretrained_params["pred_dropout"],
                    activation=pretrained_params["pred_activation"],
                    n_tasks=num_endpoints,
                    criterion=ChempropLossRegistry[loss_fn](),
                )
                # Filter out final output layer keys — their shapes depend on
                # n_tasks and will mismatch when num_endpoints differs from the
                # pretrained model. strict=False only handles missing/unexpected
                # keys, not shape mismatches.
                final_layer_prefix = f"{len(new_predictor.ffn) - 1}."
                filtered_state_dict = {
                    k: v
                    for k, v in old_state_dict.items()
                    if not k.startswith(final_layer_prefix)
                }
                # Transfer compatible weights (hidden layers share shapes)
                incompatible = new_predictor.ffn.load_state_dict(
                    filtered_state_dict, strict=False
                )
                if incompatible.missing_keys:
                    # Filter out final output layer keys — those are expected to
                    # differ when num_endpoints changes
                    final_layer_prefix = f"{len(new_predictor.ffn) - 1}."
                    unexpected_missing = [
                        k
                        for k in incompatible.missing_keys
                        if not k.startswith(final_layer_prefix)
                    ]
                    if unexpected_missing:
                        raise RuntimeError(
                            f"Failed to transfer pretrained weights. "
                            f"Missing keys: {unexpected_missing}"
                        )
                self.predictor = new_predictor
                self.metrics = nn.ModuleList(
                    [self.predictor._T_default_metric(), self.criterion.clone()]
                )

        else:
            # Case D: Full checkpoint, hidden dims specified
            # → replace predictor entirely with new FFN of specified dimensions
            self.predictor = TargetFFN(
                input_dim=self.predictor.ffn.input_dim,
                hidden_dim=pred_hidden_dim,
                n_layers=pred_num_layers,
                dropout=pred_dropout,
                activation=pred_activation,
                n_tasks=num_endpoints,
                criterion=ChempropLossRegistry[loss_fn](),
            )
            self.metrics = nn.ModuleList(
                [self.predictor._T_default_metric(), self.criterion.clone()]
            )

        self._label_names = []

    def _build_pretrain_config(self) -> dict:
        """Capture metadata needed to reconstruct the module graph at load time.

        This is called at save time to embed the config in the checkpoint.
        """
        return {
            "origin_type": "chemprop",
            "pretrain_params": self.hparams["pretrain_params"],
        }
