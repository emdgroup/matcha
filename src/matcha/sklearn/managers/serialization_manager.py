import datetime
import inspect
import os

import torch

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.torch.models.finetuning.chemprop_finetuner import ChempropFinetuner
from matcha.torch.models.finetuning.finetuner import (
    Finetuner,
    _SELF_CONTAINED_SENTINEL,
)
from matcha.utils import (
    load_pickle,
    load_yaml,
    save_pickle,
    save_yaml,
)
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.sklearn_api import (
    ScikitLearnInputModel,
    MetadataInputModel,
)
from matcha import __version__

# Canonical file names for the YAML-based serialization layout
_CONFIG_DIR = "config"
_STATE_DIR = "state"
_MODEL_YAML = "model.yaml"
_TRAINING_YAML = "training.yaml"
_DATAMODULE_YAML = "datamodule.yaml"
_METADATA_YAML = "metadata.yaml"
_CALIBRATION_YAML = "calibration.yaml"
_TUNING_YAML = "tuning.yaml"
_MLFLOW_YAML = "mlflow.yaml"
_MANIFEST_YAML = "manifest.yaml"
_PRETRAIN_CONFIG_YAML = "pretrain_config.yaml"
_CHECKPOINT_FILE = "model.ckpt"
_DATAMODULE_STATE_FILE = "datamodule_state.pkl"
_CALIBRATOR_FILE = "calibrator.pkl"


class SerializationManager:
    """Handles saving, loading, and exporting sklearn model artifacts.

    Artifact layout::

        <folder_path>/
        ├── model.ckpt                     # Lightning checkpoint (weights only)
        ├── config/
        │   ├── manifest.yaml              # Serialization version & class info
        │   ├── model.yaml                 # Model architecture params
        │   ├── training.yaml              # Training params
        │   ├── datamodule.yaml            # Datamodule params
        │   ├── metadata.yaml              # Model metadata
        │   ├── calibration.yaml           # Calibration params (optional)
        │   ├── tuning.yaml               # HPO params (optional)
        │   └── mlflow.yaml               # MLflow params (optional)
        └── state/
            ├── datamodule_state.pkl       # Fitted scalers, encoders, transforms
            └── calibrator.pkl             # Fitted calibrator object (optional)
    """

    def __init__(self):
        self.logger = get_default_logger("SERIALIZATION")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, model_instance, folder_path: str, quantize: bool = False):
        """Save all components of a sklearn model into the target folder.

        :param model_instance: the sklearn model instance to save
        :param str folder_path: root folder where artifacts are stored
        :param bool quantize: whether to quantize the model before saving
        """
        self.logger.info("Serialization: saving model and metadata")

        config_dir = os.path.join(folder_path, _CONFIG_DIR)
        state_dir = os.path.join(folder_path, _STATE_DIR)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        # --- Checkpoint ---------------------------------------------------
        if quantize:
            self.quantize(model_instance)
            torch.save(
                {
                    "state_dict": model_instance._model.state_dict(),
                    "hyper_parameters": model_instance._model.hparams,
                },
                os.path.join(folder_path, _CHECKPOINT_FILE),
            )
        else:
            model_instance._training_manager.save_checkpoint(
                os.path.join(folder_path, _CHECKPOINT_FILE)
            )

        # --- Make Finetuner checkpoints self-contained --------------------
        self._make_finetuner_self_contained(model_instance, folder_path)

        # --- Config YAMLs ------------------------------------------------
        params = model_instance.params

        save_yaml(
            os.path.join(config_dir, _MODEL_YAML),
            params.model.model_dump(),
        )
        save_yaml(
            os.path.join(config_dir, _TRAINING_YAML),
            params.training.model_dump(),
        )
        save_yaml(
            os.path.join(config_dir, _DATAMODULE_YAML),
            params.datamodule.model_dump(),
        )
        save_yaml(
            os.path.join(config_dir, _METADATA_YAML),
            params.metadata.model_dump(),
        )

        if params.calibration is not None:
            save_yaml(
                os.path.join(config_dir, _CALIBRATION_YAML),
                params.calibration.model_dump(),
            )

        if params.tuning is not None:
            save_yaml(
                os.path.join(config_dir, _TUNING_YAML),
                params.tuning.model_dump(),
            )

        if params.mlflow is not None:
            save_yaml(
                os.path.join(config_dir, _MLFLOW_YAML),
                params.mlflow.model_dump(),
            )

        # Manifest for quick identification
        save_yaml(
            os.path.join(config_dir, _MANIFEST_YAML),
            {
                "matcha_version": __version__,
                "serialization_version": 2,
                "origin_type": "classic",
                "class_name": model_instance.__class__.__name__,
                "task_type": model_instance.task_type,
                "saved_at": datetime.datetime.now().isoformat(),
            },
        )

        # --- Stateful objects (pickle) ------------------------------------
        datamodule_state = model_instance.datamodule.state_dict()
        save_pickle(
            os.path.join(state_dir, _DATAMODULE_STATE_FILE),
            datamodule_state,
        )

        if model_instance._uncertainty_manager.calibrator is not None:
            save_pickle(
                os.path.join(state_dir, _CALIBRATOR_FILE),
                model_instance._uncertainty_manager.calibrator,
            )

        self.logger.info("Serialization: artifacts saved successfully")

    def _make_finetuner_self_contained(self, model_instance, folder_path: str) -> None:
        """Post-process a Finetuner checkpoint to be self-contained.

        Embeds `_pretrain_config` into the checkpoint's hyper_parameters and
        replaces `path_to_pretrained` with the sentinel value so that future
        loads do not require access to ancestor model paths.

        Also writes `pretrain_config.yaml` to the config directory for
        human-readable redundancy.
        """
        if not isinstance(model_instance._model, (Finetuner, ChempropFinetuner)):
            return

        ckpt_path = os.path.join(folder_path, _CHECKPOINT_FILE)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        pretrain_config = model_instance._model._build_pretrain_config()

        # Inject into checkpoint hyper_parameters
        hparams = ckpt["hyper_parameters"]
        hparams["_pretrain_config"] = pretrain_config
        hparams["path_to_pretrained"] = _SELF_CONTAINED_SENTINEL

        torch.save(ckpt, ckpt_path)

        # Update the model's params so model.yaml (written later) also
        # contains the sentinel instead of the original filesystem path.
        model_instance._model.params.path_to_pretrained = _SELF_CONTAINED_SENTINEL

        # Write human-readable copy to config directory
        config_dir = os.path.join(folder_path, _CONFIG_DIR)
        save_yaml(
            os.path.join(config_dir, _PRETRAIN_CONFIG_YAML),
            pretrain_config,
        )

        self.logger.info("Finetuner checkpoint made self-contained (sentinel embedded)")

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(self, model_instance):
        """Apply dynamic quantization to the model.

        :param model_instance: the sklearn model instance to quantize
        """
        pytorch_model = model_instance._training_manager.trainer.model.cpu()
        pytorch_model.eval()

        quantized_model = torch.quantization.quantize_dynamic(
            pytorch_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

        model_instance._model = quantized_model

    # ------------------------------------------------------------------
    # get_input_args
    # ------------------------------------------------------------------

    def get_input_args(self, model_instance) -> dict:
        """Return the input arguments for the model as a dictionary.

        Merges datamodule, model, and training params, then filters to only
        those that appear in the child class constructor signature.

        :param model_instance: the sklearn model instance
        :return dict: dictionary of constructor arguments
        """
        signature = inspect.signature(model_instance.__class__.__init__)
        param_names = set(signature.parameters.keys())
        param_names.discard("self")

        all_params = {}
        if model_instance.params.datamodule:
            all_params.update(model_instance.params.datamodule.model_dump())
        if model_instance.params.model:
            all_params.update(model_instance.params.model.model_dump())
        if model_instance.params.training:
            all_params.update(model_instance.params.training.model_dump())

        return {k: v for k, v in all_params.items() if k in param_names}

    # ------------------------------------------------------------------
    # export_to_yaml  (Option A: writes the config/ subtree only)
    # ------------------------------------------------------------------

    def export_to_yaml(self, model_instance, path: str) -> None:
        """Export the model configuration YAMLs to a directory (no weights, no state).

        Writes the same ``config/`` subtree that :meth:`save` produces, so the
        output can be inspected, version-controlled, or used as a template.

        :param model_instance: the sklearn model instance
        :param str path: directory where the config YAMLs will be written
        """
        config_dir = os.path.join(path, _CONFIG_DIR)
        os.makedirs(config_dir, exist_ok=True)

        params = model_instance.params

        save_yaml(os.path.join(config_dir, _MODEL_YAML), params.model.model_dump())
        save_yaml(
            os.path.join(config_dir, _TRAINING_YAML), params.training.model_dump()
        )
        save_yaml(
            os.path.join(config_dir, _DATAMODULE_YAML), params.datamodule.model_dump()
        )
        save_yaml(
            os.path.join(config_dir, _METADATA_YAML), params.metadata.model_dump()
        )

        if params.calibration is not None:
            save_yaml(
                os.path.join(config_dir, _CALIBRATION_YAML),
                params.calibration.model_dump(),
            )
        if params.tuning is not None:
            save_yaml(
                os.path.join(config_dir, _TUNING_YAML), params.tuning.model_dump()
            )
        if params.mlflow is not None:
            save_yaml(
                os.path.join(config_dir, _MLFLOW_YAML), params.mlflow.model_dump()
            )

        save_yaml(
            os.path.join(config_dir, _MANIFEST_YAML),
            {
                "matcha_version": __version__,
                "serialization_version": 2,
                "origin_type": "classic",
                "class_name": model_instance.__class__.__name__,
                "task_type": model_instance.task_type,
                "saved_at": datetime.datetime.now().isoformat(),
            },
        )

        self.logger.info(f"Config YAMLs exported to {config_dir}")

    # ------------------------------------------------------------------
    # Load from folder
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_folder(cls, folder_path: str, accelerator: str = "cuda"):
        """Reconstruct a fully restored copy of a saved sklearn model.

        :param cls: the model class to instantiate
        :param str folder_path: location of folder with saved model artifacts
        :param str accelerator: accelerator to use for model loading
        :return: fully restored class instance
        """
        logger = get_default_logger("SERIALIZATION")

        config_dir = os.path.join(folder_path, _CONFIG_DIR)
        state_dir = os.path.join(folder_path, _STATE_DIR)

        # --- Load config YAMLs -------------------------------------------
        model_params = load_yaml(os.path.join(config_dir, _MODEL_YAML))
        training_params = load_yaml(os.path.join(config_dir, _TRAINING_YAML))
        datamodule_params = load_yaml(os.path.join(config_dir, _DATAMODULE_YAML))
        metadata = load_yaml(os.path.join(config_dir, _METADATA_YAML))

        calibration_path = os.path.join(config_dir, _CALIBRATION_YAML)
        calibration_params = (
            load_yaml(calibration_path) if os.path.exists(calibration_path) else None
        )

        tuning_path = os.path.join(config_dir, _TUNING_YAML)
        tuning_params = load_yaml(tuning_path) if os.path.exists(tuning_path) else None

        mlflow_path = os.path.join(config_dir, _MLFLOW_YAML)
        mlflow_params = load_yaml(mlflow_path) if os.path.exists(mlflow_path) else None

        # Override accelerator for current loading context
        training_params["accelerator"] = accelerator

        # --- Reconstruct full param dict ----------------------------------
        param_dict = {
            "model": model_params,
            "training": training_params,
            "datamodule": datamodule_params,
            "metadata": metadata,
            "calibration": calibration_params,
            "tuning": tuning_params,
            "mlflow": mlflow_params,
        }

        # --- Create class shell -------------------------------------------
        class_instance = cls.from_config(param_dict)

        # For self-contained chemprop checkpoints, correct the architecture
        # since reload_pretrained cannot detect torch_type in sentinel mode.
        if model_params.get(
            "path_to_pretrained"
        ) == _SELF_CONTAINED_SENTINEL and "chemprop" in model_params.get(
            "torch_type", ""
        ):
            class_instance._architecture = ChempropFinetuner

        # Restore canonical state from the loaded YAML params
        validated = ScikitLearnInputModel.model_validate(
            {**param_dict, "task_type": class_instance.task_type}
        )
        class_instance._training_manager.configure(validated.training.model_dump())
        class_instance._metadata = validated.metadata

        # --- Load model weights -------------------------------------------
        ckpt_path = os.path.join(folder_path, _CHECKPOINT_FILE)
        try:
            class_instance._model = class_instance._architecture.load_from_checkpoint(
                ckpt_path, map_location=accelerator
            )
        except Exception:
            logger.warning("Lightning loading failed, using torch loading...")
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=accelerator)
            state_dict = ckpt.get("state_dict", ckpt)
            filtered_model_params = model_params.copy()
            filtered_model_params.pop("torch_type", None)

            # For self-contained Finetuner checkpoints, pass the embedded
            # _pretrain_config so skeleton reconstruction works.
            ckpt_hparams = ckpt.get("hyper_parameters", {})
            if (
                filtered_model_params.get("path_to_pretrained")
                == _SELF_CONTAINED_SENTINEL
                and "_pretrain_config" in ckpt_hparams
            ):
                filtered_model_params["_pretrain_config"] = ckpt_hparams[
                    "_pretrain_config"
                ]

            model = class_instance._architecture(**filtered_model_params)
            model.load_state_dict(state_dict, strict=False)
            class_instance._model = model

        # --- Load datamodule state ----------------------------------------
        dm_state_path = os.path.join(state_dir, _DATAMODULE_STATE_FILE)
        datamodule_state = load_pickle(dm_state_path)
        class_instance._datamodule_manager.datamodule = DataModuleRegistry[
            datamodule_state["ID"]
        ].dummy()
        class_instance._datamodule_manager.datamodule.load_state_dict(datamodule_state)

        # --- Load calibrator if present -----------------------------------
        calibrator_pkl_path = os.path.join(state_dir, _CALIBRATOR_FILE)
        if os.path.exists(calibrator_pkl_path):
            class_instance._uncertainty_manager._calibrator = load_pickle(
                calibrator_pkl_path
            )

        # Mark as fitted since we loaded trained weights
        class_instance._training_manager._is_fitted = True

        return class_instance

    # ------------------------------------------------------------------
    # Load from config (create class shell without fitted weights)
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_config(cls, param_dict: dict):
        """Create a new class instance 'shell' from saved params.

        Expects a structured dict with top-level keys:
        ``model``, ``training``, ``datamodule``, ``metadata``,
        and optionally ``calibration``, ``tuning``.

        :param cls: the model class to instantiate
        :param dict param_dict: structured parameter dictionary
        :return: class instance with architecture but no fitted weights
        """
        param_dict = param_dict.copy()

        # Get the class's __init__ signature to filter relevant parameters
        init_signature = inspect.signature(cls.__init__)
        valid_params = set(init_signature.parameters.keys())
        valid_params.discard("self")

        # Merge the three parameter groups that map to constructor args
        all_params = {}
        all_params.update(param_dict.get("datamodule", {}))
        all_params.update(param_dict.get("model", {}))
        all_params.update(param_dict.get("training", {}))

        pruned_params = {k: v for k, v in all_params.items() if k in valid_params}

        # Create instance with filtered parameters
        class_instance = cls(**pruned_params)

        # Restore metadata from saved params
        metadata = param_dict["metadata"]
        class_instance._metadata = MetadataInputModel(**metadata)

        # Restore training params from saved params (training manager owns these)
        if "training" in param_dict:
            class_instance._training_manager.configure(param_dict["training"])

        return class_instance
