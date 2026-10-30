import datetime
import os

import yaml

from matcha import __version__
from matcha.utils.logging import get_default_logger
from matcha.utils.serialization import (
    load_pickle,
    load_yaml,
    save_pickle,
    save_yaml,
    _sanitize_for_yaml,
)
from matcha.utils.schemas.calibration import CalibratorModel
from matcha.utils.schemas.sklearn_api import MetadataInputModel

# Canonical file names – mirrors the single-model serialization layout
_CONFIG_DIR = "config"
_STATE_DIR = "state"
_MANIFEST_YAML = "manifest.yaml"
_ENSEMBLE_YAML = "ensemble.yaml"
_METADATA_YAML = "metadata.yaml"
_CALIBRATION_YAML = "calibration.yaml"
_MLFLOW_YAML = "mlflow.yaml"
_CALIBRATOR_FILE = "calibrator.pkl"


class EnsembleSerializationManager:
    """Handles saving, loading, and exporting ensemble model artifacts.

    Artifact layout::

        <path>/
        ├── config/
        │   ├── manifest.yaml          # Serialization version & class info
        │   ├── ensemble.yaml          # Ensemble-specific params (architecture, n_models, seed, learner)
        │   ├── metadata.yaml          # Model metadata
        │   ├── calibration.yaml       # Calibration params (optional)
        │   └── mlflow.yaml            # MLflow params (optional)
        ├── state/
        │   └── calibrator.pkl         # Fitted calibrator object (optional)
        ├── model_0/                   # First member (saved by its own serialization)
        │   ├── model.ckpt
        │   ├── config/
        │   └── state/
        ├── model_1/
        │   └── ...
        └── model_N/
            └── ...
    """

    def __init__(self):
        self.logger = get_default_logger("ENSEMBLE_SERIALIZATION")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, ensemble, path: str, quantize: bool = False) -> None:
        """Save all members and ensemble-level artifacts to *path*.

        :param ensemble: the Ensemble instance to save
        :param str path: root folder where artifacts are stored
        :param bool quantize: whether to quantize models before saving
        """
        self.logger.info(f"Saving ensemble to {path}")
        ensemble._sync_params()

        config_dir = os.path.join(path, _CONFIG_DIR)
        state_dir = os.path.join(path, _STATE_DIR)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        # --- Save individual members --------------------------------------
        for i, model in enumerate(ensemble._model_box):
            model.save_model(f"{path}/model_{i}", quantize)

        # --- Config YAMLs ------------------------------------------------
        params = ensemble.params

        # Manifest for quick identification (mirrors single-model pattern)
        save_yaml(
            os.path.join(config_dir, _MANIFEST_YAML),
            {
                "matcha_version": __version__,
                "serialization_version": 2,
                "class_name": "Ensemble",
                "task_type": ensemble.task_type,
                "saved_at": datetime.datetime.now().isoformat(),
            },
        )

        # Ensemble-specific params
        save_yaml(
            os.path.join(config_dir, _ENSEMBLE_YAML),
            {
                "architecture": params.architecture,
                "n_models": params.n_models,
                "seed": params.seed,
                "learner": params.learner.model_dump() if params.learner else None,
            },
        )

        # Metadata
        save_yaml(
            os.path.join(config_dir, _METADATA_YAML),
            params.metadata.model_dump(),
        )

        # Calibration (optional)
        if params.calibration is not None:
            save_yaml(
                os.path.join(config_dir, _CALIBRATION_YAML),
                params.calibration.model_dump(),
            )

        # MLflow (optional)
        if params.mlflow is not None:
            save_yaml(
                os.path.join(config_dir, _MLFLOW_YAML),
                params.mlflow.model_dump(),
            )

        # --- Stateful objects (pickle) ------------------------------------
        save_pickle(
            os.path.join(state_dir, _CALIBRATOR_FILE),
            ensemble.calibrator,
        )

        self.logger.info("Ensemble saved successfully")

    # ------------------------------------------------------------------
    # Export to YAML
    # ------------------------------------------------------------------

    def export_to_yaml(self, ensemble, path: str) -> None:
        """Export the ensemble configuration and parameters to a YAML file.

        Uses the first model's input args as the learner config and adds
        ensemble-specific fields (architecture, metadata, n_models).

        :param ensemble: the Ensemble instance
        :param str path: file path for the YAML output
        """
        learner_params = ensemble._model_box[0].get_input_args()
        output = {
            "model": {
                "params": learner_params,
                "architecture": ensemble._architecture_name,
                "metadata": {
                    "model_name": ensemble.params.metadata.model_name,
                    "model_owner": ensemble.params.metadata.model_owner,
                    "model_scope": ensemble.params.metadata.model_scope,
                    "model_version": ensemble.params.metadata.model_version,
                    "description": ensemble.params.metadata.description,
                },
                "ensemble": ensemble.params.n_models,
            }
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.logger.info(f"Saving model config as YAML at {path}")
        with open(path, "w") as f:
            yaml.dump(_sanitize_for_yaml(output), f)
        self.logger.warning("Please remember to fill in missing metadata info")
        self.logger.warning(
            "Fields to fill: model_owner, model_scope, model_name, model_version"
        )

    # ------------------------------------------------------------------
    # Load from config
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_config(cls, params: dict):
        """Reconstruct an Ensemble from a saved parameter dictionary.

        Creates individual model instances using the registered architecture's
        ``from_config``, then wraps the first one into a new Ensemble via the
        standard ``__init__``.

        :param cls: the Ensemble class
        :param dict params: saved ensemble parameters (structured format with
            ``ensemble``, ``metadata``, ``calibration`` keys)
        :return: reconstructed Ensemble instance
        """
        from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry

        ens_cfg = params["ensemble"]
        architecture = ens_cfg["architecture"]
        learner = ens_cfg.get("learner")
        n_models = ens_cfg["n_models"]
        seed = ens_cfg.get("seed", 0)

        # reconstruct the template model from the structured learner config
        model_cls = ScikitLearnModelRegistry[architecture]
        if isinstance(learner, dict) and all(
            key in learner for key in ["training", "model", "datamodule"]
        ):
            template = model_cls.from_config(learner)
        else:
            template = model_cls(**learner)

        class_instance = cls(model=template, n_models=n_models, seed=seed)

        metadata = params.get("metadata")
        if metadata is not None:
            class_instance.params.metadata = MetadataInputModel(**metadata)

        calibration = params.get("calibration")
        if calibration is not None:
            class_instance.params.calibration = CalibratorModel(**calibration)
            class_instance._calibration_manager.create_calibrator(
                config=class_instance.params.calibration
            )

        class_instance._sync_params()
        return class_instance

    # ------------------------------------------------------------------
    # Load from folder
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_folder(cls, path: str, accelerator: str = "cuda"):
        """Load a saved Ensemble from a folder.

        Each member is individually loaded via its architecture's
        ``from_folder``, then assembled into an Ensemble shell.

        :param cls: the Ensemble class
        :param str path: folder containing ensemble artifacts
        :param str accelerator: device to load models onto
        :return: fully restored Ensemble instance
        """
        from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry

        logger = get_default_logger("ENSEMBLE_SERIALIZATION")

        config_dir = os.path.join(path, _CONFIG_DIR)
        state_dir = os.path.join(path, _STATE_DIR)

        # --- Load config YAMLs -------------------------------------------
        ensemble_params = load_yaml(os.path.join(config_dir, _ENSEMBLE_YAML))
        metadata = load_yaml(os.path.join(config_dir, _METADATA_YAML))

        calibration_path = os.path.join(config_dir, _CALIBRATION_YAML)
        calibration_params = (
            load_yaml(calibration_path) if os.path.exists(calibration_path) else None
        )

        # --- Load stateful objects ----------------------------------------
        calibrator_pkl_path = os.path.join(state_dir, _CALIBRATOR_FILE)
        calibrator = (
            load_pickle(calibrator_pkl_path)
            if os.path.exists(calibrator_pkl_path)
            else None
        )

        architecture = ensemble_params["architecture"]
        n_models = ensemble_params["n_models"]
        seed = ensemble_params.get("seed", 0)
        model_cls = ScikitLearnModelRegistry[architecture]

        # load each member individually (they have fitted weights)
        model_box = [
            model_cls.from_folder(f"{path}/model_{i}", accelerator=accelerator)
            for i in range(n_models)
        ]

        # build the ensemble shell from the first loaded model
        class_instance = cls(
            model=model_box[0],
            n_models=n_models,
            seed=seed,
        )

        # replace the model box with the individually loaded (fitted) members
        class_instance._model_box = model_box
        class_instance._calibration_manager._calibrator = calibrator

        # Restore metadata
        class_instance.params.metadata = MetadataInputModel(**metadata)

        # Restore calibration params
        if calibration_params is not None:
            class_instance.params.calibration = CalibratorModel(**calibration_params)
            class_instance._calibration_manager.create_calibrator(
                config=class_instance.params.calibration
            )

        class_instance._sync_params()

        logger.info(f"Ensemble loaded from {path}")
        return class_instance
