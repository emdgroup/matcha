"""Finetuning wrappers for pretrained models via the scikit-learn-compatible API."""

import os

from lightning.pytorch.core.mixins import HyperparametersMixin
import lightning as L
from matcha.datamodules import CombinedDataModule, TabularDataModule
from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.torch.models.finetuning import ChempropFinetuner, Finetuner
from matcha.torch.models.finetuning.finetuner import _SELF_CONTAINED_SENTINEL
from matcha.sklearn.base_sklearn_model import (
    BaseScikitLearnModel,
    ScikitLearnClassifierMixin,
    ScikitLearnModelRegistry,
    ScikitLearnRegressorMixin,
)
from matcha.sklearn.managers import FinetunerTrainingManager
from matcha.utils import load_yaml, load_pickle
import numpy as np
import torch
from matcha.utils.schemas.sklearn_api import TrainingInputModel, MetadataInputModel
from rdkit.Chem.rdchem import Mol
from torch.utils.data import StackDataset


class BaseFinetuner(BaseScikitLearnModel, HyperparametersMixin):
    """Base class for finetuning pretrained encoders on downstream tasks.

    Unlike standard sklearn-API models that train from scratch, finetuners
    load a pretrained encoder (classic or pretraining-origin) and attach
    a new prediction head. The encoder can be frozen, fully finetuned,
    or adapted via LoRA depending on ``finetuning_strategy``.

    Subclasses :class:`FinetuningRegressor` and :class:`FinetuningClassifier`
    combine this base with the appropriate task mixin.
    """

    def __init__(self):
        # Reuse the canonical _start_setup() from BaseScikitLearnModel so that
        # any new managers added there are automatically picked up.
        self._start_setup()

        # Override the training manager with the finetuner-specific variant
        # (provides custom MLflow logging behaviour).
        self._training_manager = FinetunerTrainingManager()

    def _make_model_instance(self):
        """Placeholder — model creation is handled by :meth:`reload_pretrained`."""
        pass

    def _adapt_dict_for_modality(
        self, datamodule_params: dict, model_params: dict
    ) -> tuple[dict, dict]:
        """No modality-specific adaptation needed for finetuners."""
        return datamodule_params, model_params

    def _create_datamodule(self, datamodule_params: dict, train_params: dict) -> None:
        """Not used — finetuner restores datamodule from pretrained state."""
        pass

    # @classmethod
    # def from_config(cls, param_dict, create_empty=True):
    #     params = copy.deepcopy(param_dict)
    #     try:
    #         class_instance = super().from_config(params)
    #     except Exception as e:
    #         original_path = params['model']['path_to_pretrained']
    #         model_name = original_path.rstrip('/').split('/')[-1]
    #         params['model']['path_to_pretrained'] = f'/models/foundation/{model_name}'
    #         class_instance = super().from_config(params)
    #     return class_instance

    def reload_pretrained(
        self,
        path_to_pretrained: str,
        pred_hidden_dims: list[int] = [256, 256],
        task_head_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 0.0001},
        pretrain_lr: float = 1e-4,
        pretrain_decay: float = 0.5,
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {},
        finetuning_strategy: str = "full",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_min_dim: int = 32,
        keep_existing_predictor: bool = True,
        num_epochs: int = 20,
        batch_size: int = 64,
        stochastic_weight_averaging: bool = False,
        early_stopping: bool = True,
        patience: int = 10,
        devices: int = 1,
        accelerator: str = "gpu",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        scaler_type: str = "standard",
        seed: int = 0,
    ):
        """Load a pretrained model and configure the finetuning setup.

        Restores the encoder and datamodule from a saved model folder,
        creates a new prediction head, and configures training parameters
        for downstream finetuning.

        :param str path_to_pretrained: path to the saved pretrained model folder
        :param list[int] pred_hidden_dims: hidden layer dimensions for the prediction head
        :param list[int] | None task_head_dims: per-task head dimensions (multitask)
        :param str activation: activation function name for the prediction head
        :param float dropout: dropout rate for the prediction head
        :param int num_endpoints: number of output endpoints (tasks)
        :param str loss_fn: loss function name
        :param dict loss_args: additional arguments for the loss function
        :param str optimizer: optimizer name
        :param dict optimizer_args: optimizer keyword arguments
        :param float pretrain_lr: learning rate for the pretrained encoder layers
        :param float pretrain_decay: LR decay factor for the pretrained encoder
        :param str scheduler: learning rate scheduler name
        :param dict scheduler_args: scheduler keyword arguments
        :param str finetuning_strategy: one of 'full', 'frozen', or 'lora'
        :param int lora_rank: rank for LoRA adaptation
        :param float lora_alpha: scaling factor for LoRA
        :param int lora_min_dim: minimum dimension for LoRA layers
        :param bool keep_existing_predictor: whether to preserve the pretrained
            model's predictor hidden layers in the forward path, defaults to
            ``True``. When ``True``, the pretrained predictor's hidden layers
            stay in place and the new ``pred_hidden_dims`` MLP is stacked on
            top; only the pretrained model's final prediction head is dropped.
            When ``False``, the pretrained predictor is discarded end-to-end
            and the new ``pred_hidden_dims`` MLP consumes the leaf encoder's
            output directly. Not supported for Chemprop pretrained models.
        :param int num_epochs: number of training epochs
        :param int batch_size: batch size for training
        :param bool stochastic_weight_averaging: whether to apply SWA
        :param bool early_stopping: whether to use early stopping
        :param int patience: early stopping patience (epochs)
        :param int devices: number of devices to use
        :param str accelerator: hardware accelerator type
        :param bool clip: whether to clip predictions to training range
        :param dict label_encoder_params: label encoder configuration
        :param str | list[str] | dict | None label_transform_map: label transform spec
        :param str scaler_type: type of y-scaler to use
        :param int seed: random seed for reproducibility
        """
        # Self-contained mode: model and datamodule will be populated by
        # load_from_folder via load_from_checkpoint — skip filesystem loading.
        if path_to_pretrained == _SELF_CONTAINED_SENTINEL:
            self._architecture = Finetuner
            return

        # Load original params and datamodule state from YAML layout
        config_dir = os.path.join(path_to_pretrained, "config")
        state_dir = os.path.join(path_to_pretrained, "state")

        params = {
            "model": load_yaml(os.path.join(config_dir, "model.yaml")),
            "training": load_yaml(os.path.join(config_dir, "training.yaml")),
            "datamodule": load_yaml(os.path.join(config_dir, "datamodule.yaml")),
            "metadata": load_yaml(os.path.join(config_dir, "metadata.yaml")),
        }
        dm_state = load_pickle(os.path.join(state_dir, "datamodule_state.pkl"))

        # ------------------------------------------------------------------
        # Detect origin type from manifest
        # ------------------------------------------------------------------
        manifest_path = os.path.join(config_dir, "manifest.yaml")
        manifest = load_yaml(manifest_path) if os.path.exists(manifest_path) else {}
        origin_type = manifest.get("origin_type", "classic")

        # ------------------------------------------------------------------
        # Restore datamodule — pretraining DMs get converted to classic
        # ------------------------------------------------------------------
        if origin_type == "pretraining":
            from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
                GraphPretrainingDataModule,
            )
            from matcha.datamodules.pretraining.clm_mlm_datamodule import (
                CLMMLMDataModule,
            )

            raw_dm = DataModuleRegistry[dm_state["ID"]].dummy()
            raw_dm.load_state_dict(dm_state)

            if isinstance(raw_dm, GraphPretrainingDataModule):
                datamodule = raw_dm.export_to_classic()
            elif isinstance(raw_dm, CLMMLMDataModule):
                datamodule = raw_dm.export_to_classic()
            else:
                # Fallback: try export_to_classic if available
                if hasattr(raw_dm, "export_to_classic"):
                    datamodule = raw_dm.export_to_classic()
                else:
                    datamodule = raw_dm
        else:
            datamodule = DataModuleRegistry[dm_state["ID"]].dummy()
            datamodule.load_state_dict(dm_state)

        # Overwrite objects in datamodule
        datamodule._create_y_scaler(scaler_type)
        datamodule._create_label_encoder(label_encoder_params)
        datamodule._create_label_transform({"transform_map": label_transform_map})
        datamodule.params.clip = clip

        # Freeze tabular datamodule(s) robustly
        if isinstance(datamodule, CombinedDataModule):
            for sub_dm in getattr(datamodule, "_datamodules", []):
                if isinstance(sub_dm, TabularDataModule):
                    sub_dm._freeze_x_scaler = True
        elif isinstance(datamodule, TabularDataModule):
            datamodule._freeze_x_scaler = True

        # Save original model params for metadata
        original_model_params = params.get("model", {})

        # Create training info
        training = TrainingInputModel(
            num_epochs=num_epochs,
            batch_size=batch_size,
            stochastic_weight_averaging=stochastic_weight_averaging,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            patience=patience,
            seed=seed,
        )

        # Set seeds before object creation
        L.seed_everything(seed=training.seed, workers=True, verbose=False)
        self.logger.info(f"Setting seed {training.seed} for model instance")

        # Create new model
        if origin_type == "pretraining":
            # For pretraining models, the Finetuner loads encoder-only
            # via PretrainedEncoderWrapper.  The architecture string is
            # set to "finetunermodel" because the Finetuner init branch
            # for pretraining is triggered by the manifest, not the
            # architecture string.
            model = Finetuner(
                architecture=params["model"]["torch_type"],
                path_to_pretrained=path_to_pretrained,
                pred_hidden_dims=pred_hidden_dims,
                task_head_dims=task_head_dims,
                activation=activation,
                dropout=dropout,
                num_endpoints=num_endpoints,
                loss_fn=loss_fn,
                loss_args=loss_args,
                optimizer=optimizer,
                optimizer_args=optimizer_args,
                pretrain_lr=pretrain_lr,
                pretrain_decay=pretrain_decay,
                scheduler=scheduler,
                scheduler_args=scheduler_args,
                finetuning_strategy=finetuning_strategy,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_min_dim=lora_min_dim,
                keep_existing_predictor=keep_existing_predictor,
            )
            self._architecture = Finetuner
        elif "chemprop" in params["model"]["torch_type"]:
            # ChempropFinetuner has no notion of a stripped-vs-kept pretrained
            # predictor, so reject any non-default value explicitly rather
            # than silently dropping the flag.
            if not keep_existing_predictor:
                raise ValueError(
                    "keep_existing_predictor=False is not supported for "
                    "Chemprop pretrained models; ChempropFinetuner does not "
                    "expose a stripped-predictor mode."
                )

            # Chemprop uses its own NoamLR schedule — silently override
            # any non-chemprop scheduler the caller may have left at the
            # default.  Also replace scheduler_args when they don't carry
            # the keys that ChempropFinetuner expects.
            scheduler = "chemprop"
            if "max_lr" not in scheduler_args:
                scheduler_args = {"warmup_epochs": 5, "max_lr": 1e-4, "final_lr": 1e-5}

            model = ChempropFinetuner(
                path_to_pretrained,
                num_endpoints=num_endpoints,
                loss_fn=loss_fn,
                optimizer_args=optimizer_args,
                scheduler_args=scheduler_args,
                pred_hidden_dim=pred_hidden_dims[0]
                if isinstance(pred_hidden_dims, list)
                else None,
                pred_num_layers=len(pred_hidden_dims)
                if isinstance(pred_hidden_dims, list)
                else 1,
                pred_dropout=dropout,
                pred_activation=activation,
            )
            self._architecture = ChempropFinetuner
        else:
            model = Finetuner(
                architecture=params["model"]["torch_type"] + "model",
                path_to_pretrained=path_to_pretrained,
                pred_hidden_dims=pred_hidden_dims,
                task_head_dims=task_head_dims,
                activation=activation,
                dropout=dropout,
                num_endpoints=num_endpoints,
                loss_fn=loss_fn,
                loss_args=loss_args,
                optimizer=optimizer,
                optimizer_args=optimizer_args,
                pretrain_lr=pretrain_lr,
                pretrain_decay=pretrain_decay,
                scheduler=scheduler,
                scheduler_args=scheduler_args,
                finetuning_strategy=finetuning_strategy,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_min_dim=lora_min_dim,
                keep_existing_predictor=keep_existing_predictor,
            )
            self._architecture = Finetuner

        # Update datamodule and model
        self._datamodule_manager.datamodule = datamodule
        self._model = model

        metadata = self._init_metadata()
        self._metadata = MetadataInputModel(**metadata)
        self._training_manager.configure(training.model_dump())

        # Use parent's annotate method to store original model params
        self.annotate("pretrained_params", original_model_params)

    def _inner_predict(
        self,
        x: list[Mol] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ):
        """Run inference and average over augmentations if present.

        Extends the base ``_inner_predict`` to handle CLM-style SMILES
        augmentations by averaging predictions across augmented copies.

        :param list[Mol] | StackDataset x: input to compute predictions for
        :param str | None accelerator: hardware to use for predictions
        :param int | None devices: number of devices to use
        :param int | None batch_size: batch size to use
        :returns: model predictions as a torch tensor
        :rtype: torch.Tensor
        """
        preds = super()._inner_predict(x, accelerator, devices, batch_size)

        if len(preds) == len(x):
            return preds
        else:
            n_originals = len(x)
            num_augmentations = (preds.shape[0] // n_originals) - 1

            # Extract original predictions (first N samples)
            original_preds = preds[:n_originals]

            # Extract and stack augmented predictions
            # Each augmentation has n_originals samples
            augmented_preds = []
            for aug_idx in range(num_augmentations):
                start_idx = n_originals * (1 + aug_idx)
                end_idx = start_idx + n_originals
                augmented_preds.append(preds[start_idx:end_idx])

            # Stack all predictions: (num_augmentations + 1, n_originals, num_outputs)
            all_preds = torch.stack([original_preds] + augmented_preds, dim=0)

            # Average across augmentations (dim=0)
            averaged_output = all_preds.mean(dim=0)

            return averaged_output


@ScikitLearnModelRegistry.register()
class FinetuningRegressor(BaseFinetuner, ScikitLearnRegressorMixin):
    """Regression finetuner that loads a pretrained encoder for downstream regression tasks.

    Combines :class:`BaseFinetuner` with :class:`ScikitLearnRegressorMixin`
    to provide a complete sklearn-compatible regression interface on top of
    a pretrained model.
    """

    def __init__(
        self,
        path_to_pretrained: str,
        pred_hidden_dims: list[int] = [256, 256],
        task_head_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 0.0001},
        pretrain_lr: float = 1e-4,
        pretrain_decay: float = 0.5,
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {},
        finetuning_strategy: str = "full",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_min_dim: int = 32,
        keep_existing_predictor: bool = True,
        num_epochs: int = 20,
        batch_size: int = 64,
        stochastic_weight_averaging: bool = False,
        early_stopping: bool = True,
        patience: int = 10,
        devices: int = 1,
        accelerator: str = "gpu",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        scaler_type: str = "standard",
        seed: int = 0,
    ):
        super().__init__()
        super().reload_pretrained(
            path_to_pretrained=path_to_pretrained,
            pred_hidden_dims=pred_hidden_dims,
            task_head_dims=task_head_dims,
            activation=activation,
            dropout=dropout,
            num_endpoints=num_endpoints,
            loss_fn=loss_fn,
            loss_args=loss_args,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            pretrain_lr=pretrain_lr,
            pretrain_decay=pretrain_decay,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            finetuning_strategy=finetuning_strategy,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_min_dim=lora_min_dim,
            keep_existing_predictor=keep_existing_predictor,
            num_epochs=num_epochs,
            batch_size=batch_size,
            stochastic_weight_averaging=stochastic_weight_averaging,
            early_stopping=early_stopping,
            patience=patience,
            devices=devices,
            accelerator=accelerator,
            clip=clip,
            label_encoder_params=label_encoder_params,
            label_transform_map=label_transform_map,
            scaler_type=scaler_type,
            seed=seed,
        )
        if path_to_pretrained != _SELF_CONTAINED_SENTINEL:
            self.datamodule.params.is_classification = False


@ScikitLearnModelRegistry.register()
class FinetuningClassifier(BaseFinetuner, ScikitLearnClassifierMixin):
    """Classification finetuner that loads a pretrained encoder for downstream classification.

    Combines :class:`BaseFinetuner` with :class:`ScikitLearnClassifierMixin`
    to provide a complete sklearn-compatible classification interface on top of
    a pretrained model.
    """

    def __init__(
        self,
        path_to_pretrained: str,
        pred_hidden_dims: list[int] = [256, 256],
        task_head_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
        num_endpoints: int = 1,
        loss_fn: str = "bce",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 0.0001},
        pretrain_lr: float = 1e-4,
        pretrain_decay: float = 0.5,
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {},
        finetuning_strategy: str = "full",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_min_dim: int = 32,
        keep_existing_predictor: bool = True,
        num_epochs: int = 20,
        batch_size: int = 64,
        stochastic_weight_averaging: bool = False,
        early_stopping: bool = True,
        patience: int = 10,
        devices: int = 1,
        accelerator: str = "gpu",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        scaler_type: str = "standard",
        seed: int = 0,
    ):
        # Ensure the label encoder is created as binary_classification type
        # so that user-provided class_thresholds and class_labels are preserved.
        label_encoder_params = label_encoder_params.copy()
        if "encoder_type" not in label_encoder_params:
            label_encoder_params["encoder_type"] = "binary_classification"

        super().__init__()
        super().reload_pretrained(
            path_to_pretrained=path_to_pretrained,
            pred_hidden_dims=pred_hidden_dims,
            task_head_dims=task_head_dims,
            activation=activation,
            dropout=dropout,
            num_endpoints=num_endpoints,
            loss_fn=loss_fn,
            loss_args=loss_args,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            pretrain_lr=pretrain_lr,
            pretrain_decay=pretrain_decay,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            finetuning_strategy=finetuning_strategy,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_min_dim=lora_min_dim,
            keep_existing_predictor=keep_existing_predictor,
            num_epochs=num_epochs,
            batch_size=batch_size,
            stochastic_weight_averaging=stochastic_weight_averaging,
            early_stopping=early_stopping,
            patience=patience,
            devices=devices,
            accelerator=accelerator,
            clip=clip,
            label_encoder_params=label_encoder_params,
            label_transform_map=label_transform_map,
            scaler_type=scaler_type,
            seed=seed,
        )
        if path_to_pretrained != _SELF_CONTAINED_SENTINEL:
            self.datamodule.params.is_classification = True
            self.datamodule.params.clip = False

    def predict_proba(
        self,
        x,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Allows to return probabilities instead of class labels for classification
        models.

        :param list[Mol] | StackDataset x: input to compute predictions for

        :param str | None accelerator: hardware to use for predictions, if None
            it is kept as training settings, defaults to None

        :param int | None devices: how many resources to use, if None
            it is kept as training settings, defaults to None

        :param int | None batch_size: batch size to use, if None
            it is kept as training settings, defaults to None

        :return np.ndarray: class probabilities for the input
        """

        preds = self._inner_predict(x, accelerator, devices, batch_size)

        # NOTE: The base mixin's predict_proba checks `isinstance(self._model, ChempropModel)`.
        # Finetuners wrap a different class — ChempropFinetuner — which already applies
        # sigmoid internally, so we mirror the same logic but check for ChempropFinetuner.
        if isinstance(self._model, ChempropFinetuner):
            return preds.numpy()
        else:
            transform = torch.nn.Sigmoid()
            preds = transform(preds).numpy()
            return preds
