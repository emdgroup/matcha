"""Base classes and mixins for the scikit-learn-compatible model API."""

import datetime
from abc import ABC, abstractmethod

from matcha.torch.models.classic.chemprop_model import ChempropModel
import lightning as L
import numpy as np
import torch
from chemprop.data import MoleculeDataset
from rdkit.Chem.rdchem import Mol
from torch.utils.data import StackDataset

from matcha.datamodules.utils import CombinedStackDataset
from matcha.sklearn.managers import (
    DataModuleManager,
    MLFlowManager,
    SerializationManager,
    TrainingManager,
    UncertaintyManager,
    ExplainabilityManager,
    HPOManager,
)
from matcha.utils import silence_nuisance_warnings
from matcha.utils.logging import get_default_logger
from matcha.utils.registry import ClassRegistry
from matcha.utils.schemas.sklearn_api import (
    ScikitLearnInputModel,
    MetadataInputModel,
)
from matcha import __version__

torch.set_float32_matmul_precision("high")
# default train args
_train_args = [
    "num_epochs",
    "batch_size",
    "stochastic_weight_averaging",
    "accelerator",
    "devices",
    "early_stopping",
    "patience",
    "seed",
]

# default datamodule args
_datamodule_args = [
    "label_encoder_params",
    "label_transform_map",
    "is_classification",
    "scaler_type",
    "clip",
    "augment_resonance",
    "n_jobs_featurizer",  # general
    "max_length",
    "num_augmentations",
    "rwse_k",
    "laplacian_k",
    "elstatic_k",
    "distmat_k",
    "rrwp_k",
    "num_virtual_nodes",
    "compute_distance",  # graph/graph3d
    "feature_list",  # tabular
]


class BaseScikitLearnModel(ABC):
    """Base class for all sklearn API models. It is not meant to be instantiated directly,
    but rather to be used as a parent class for each sklearn API model.
    This class already implements all (or most) methods needed to get the API running,
    so child classes only need to parse the arguments for the desired architecture.

    **Ownership model:**

    * The ``BaseScikitLearnModel`` directly owns the Lightning model
      (``self._model``) and the run metadata (``self._metadata``).
    * All other cross-cutting concerns are delegated to dedicated managers:

      - ``DataModuleManager``       — datamodule lifecycle, featurization, label encoding
      - ``TrainingManager``         — Lightning Trainer, callbacks, training loop, fit state
      - ``MLFlowManager``           — experiment tracking and artifact logging
      - ``SerializationManager``    — save / load / export
      - ``UncertaintyManager``      — MC-dropout uncertainty and calibration
      - ``ExplainabilityManager``   — LIME-based explanations
      - ``HPOManager``              — hyperparameter optimization via Optuna

    Concrete subclasses must:

    1. Set ``self._architecture`` to the Lightning module class **before**
       calling ``super().__init__(params)``.
    2. Implement ``_create_datamodule(datamodule_params, train_params)``
       to build the appropriate datamodule.
    3. Implement ``_adapt_dict_for_modality(datamodule_params, model_params)``
       to adjust parameters for the specific modality (e.g. graph, CLM, tabular).

    :param dict params: all the arguments required to make the neural network for
        the child class work, which are then split into datamodule_params,
        model_params and train_params
    """

    #: The Lightning module class this model wraps.  Must be set by subclasses
    #: **before** ``super().__init__()`` is called.
    _architecture: type = None

    def __init__(self, params):
        """Initialize the sklearn model with the new hierarchical structure.

        The initialization follows this pattern:
        1. Setup basic attributes and managers
        2. Adapt datamodule and training dicts for task type (mixin-driven)
        3. Adapt datamodule and model dicts depending on modality
        4. Create all components using factory methods
        5. Params are assembled on demand via the params property
        """
        if self._architecture is None:
            raise TypeError(
                f"{self.__class__.__name__} must set `_architecture` before "
                f"calling super().__init__()"
            )

        self._start_setup()

        # parse arguments
        train_dict, model_dict, datamodule_dict = self._parse_args(params)

        # Adapt dicts based on whether the concrete class is a regressor or classifier
        datamodule_dict, train_dict = self._adapt_dicts_for_mixin(
            datamodule_dict, train_dict
        )

        # Adapt model_dict according to modality-specific dependencies (e.g. CLM)
        datamodule_dict, model_dict = self._adapt_dict_for_modality(
            datamodule_dict, model_dict
        )

        # Set seed before object creation
        self.logger.info(f"Setting seed {train_dict['seed']} for model instance")
        L.seed_everything(seed=train_dict["seed"], workers=True, verbose=False)

        # Configure training manager with its params
        self._training_manager.configure(train_dict)

        # Create objects via factory
        self._create_datamodule(datamodule_dict, train_dict)
        self._create_model(model_dict)

        # Create metadata
        self._metadata = MetadataInputModel(**self._init_metadata())

    def _start_setup(self):
        """Initialize basic attributes and delegate managers."""
        silence_nuisance_warnings()
        self.logger = get_default_logger("SKLEARN")
        self._model = None

        # Delegate managers
        self._datamodule_manager = DataModuleManager()
        self._mlflow_manager = MLFlowManager()
        self._serialization_manager = SerializationManager()
        self._training_manager = TrainingManager()
        self._uncertainty_manager = UncertaintyManager()
        self._explainability_manager = ExplainabilityManager()
        self._hpo_manager = HPOManager()

        self.logger.info(f"Created  {self.__class__.__name__} instance")

    @property
    def calibrator(self):
        """The calibration object used for uncertainty calibration, or None."""
        return self._uncertainty_manager.calibrator

    @property
    def explainer(self):
        """The explainability object used for LIME explanations, or None."""
        return self._explainability_manager.explainer

    @property
    def datamodule(self):
        """The Lightning datamodule managing featurization and batching."""
        return self._datamodule_manager.datamodule

    @property
    def model(self):
        """The underlying Lightning model instance."""
        return self._model

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been successfully trained."""
        return self._training_manager.is_fitted

    @property
    def is_classifier(self) -> bool:
        """Whether this model instance is a classifier (vs regressor)."""
        return isinstance(self, ScikitLearnClassifierMixin)

    @property
    def task_type(self) -> str:
        """Inferred from the mixin: 'binary_classification' or 'regression'."""
        return "binary_classification" if self.is_classifier else "regression"

    @property
    def params(self) -> ScikitLearnInputModel:
        """Assembled on demand from canonical sources — no drift possible."""
        return ScikitLearnInputModel(
            training=self._training_manager.params,
            datamodule=self._datamodule_manager.params,
            model=self._model.params,
            metadata=self._metadata,
            task_type=self.task_type,
            calibration=self._uncertainty_manager.params,
            mlflow=self._mlflow_manager.params,
            tuning=self._hpo_manager.params,
        )

    @abstractmethod
    def _create_datamodule(self, datamodule_params: dict, train_params: dict) -> None:
        """Factory method to create datamodule instance.

        :param dict datamodule_params: datamodule configuration dictionary
        :param dict train_params: training configuration dictionary
        """
        pass

    @abstractmethod
    def _adapt_dict_for_modality(
        self, datamodule_params: dict, model_params: dict
    ) -> tuple[dict]:
        """Adjust parameters for the specific modality (graph, CLM, tabular, etc.).

        :param dict datamodule_params: datamodule configuration dictionary
        :param dict model_params: model configuration dictionary
        :returns: adapted (datamodule_params, model_params) tuple
        :rtype: tuple[dict, dict]
        """
        pass

    def _create_model(self, params: dict) -> None:
        """Factory method to create a model instance according to the input.

        Creates a shallow copy to avoid mutating the caller's dictionary.

        :param dict params: model parameters
        """
        filtered = {k: v for k, v in params.items() if k != "torch_type"}
        self._model = self._architecture(**filtered)

    def _init_metadata(self):
        """Build the default metadata dictionary for a new model instance.

        :returns: metadata dictionary with placeholder values
        :rtype: dict
        """
        return {
            "model_type": self.__class__.__name__,
            "model_name": "no name provided",
            "model_version": 404,
            "model_scope": "no scope provided",
            "model_owner": "no owner provided",
            "matcha_version": __version__,
            "date": datetime.datetime.now().isoformat(),
            "description": "No description provided",
            "extra": {},
        }

    def _parse_args(
        self, args: dict, train_args=_train_args, datamodule_args=_datamodule_args
    ):
        """Utility function to parse the input args from a sklearn model into three dictionaries
        containing the training params (e.g. number of epochs), the model params (e.g. number of
        layers) and the datamodule params (e.g. which descriptors to compute).

        :param dict args: all the arguments to parse into self.datamodule_params,
            self.model_params and self.train_params

        :param list[str] train_args: names of args specific to training

        :param list[str] datamodule_args: names of args specific to datamodules
        """
        keys = list(args.keys())
        active_dm_args = [x for x in datamodule_args if x in keys]
        datamodule_dict = {
            key: value for key, value in args.items() if key in active_dm_args
        }
        train_dict = {key: value for key, value in args.items() if key in train_args}
        model_dict = {
            key: value
            for key, value in args.items()
            if key not in active_dm_args and key not in train_args
        }

        return train_dict, model_dict, datamodule_dict

    def _parse_label_transform_map(self, datamodule_args):
        """Extract label_transform_map from datamodule args and reformat it.

        Converts ``label_transform_map`` into the ``label_transform_params``
        structure expected by the datamodule.

        :param dict datamodule_args: datamodule arguments dictionary (mutated in-place)
        :returns: the updated datamodule arguments
        :rtype: dict
        """
        transform_map = datamodule_args["label_transform_map"]
        datamodule_args.pop("label_transform_map")
        datamodule_args["label_transform_params"] = {
            "transform_map": transform_map,
            "y_clip": None,
        }
        return datamodule_args

    def set_mlflow_experiment(
        self,
        experiment_name: str,
        run_name: str = "run",
        tag: dict | None = None,
        log_dir: str | None = "./matcha_log",
        server_uri: str | None = None,
    ):
        """Sets the experiment name and run name for the model.

        :param str experiment_name: name of the experiment to set
        :param str run_name: name of the run to associate with the experiment
        :param dict | None tag: tags to associate with the run
        :param str | None log_dir: directory for logging
        :param str | None server_uri: MLflow server URI
        """
        if tag is None:
            tag = {}
        self._mlflow_manager.setup_experiment(
            experiment_name=experiment_name,
            run_name=run_name,
            tag=tag,
            log_dir=log_dir,
            server_uri=server_uri,
        )

    def _inner_predict(
        self,
        x: list[Mol] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Wrapper for the forward pass of the underlying model to interface well with RDKIT
        molecule lists and return numpy arrays instead of torch tensors

        :param list[Mol] | StackDataset x: input to compute predictions for

        :param str | None accelerator: hardware to use for predictions, if None
            it is kept as training settings, defaults to None

        :param int | None devices: how many resources to use, if None
            it is kept as training settings, defaults to None

        :param int | None batch_size: batch size to use, if None
            it is kept as training settings, defaults to None

        :return torch.Tensor: model predictions as torch tensors
        """
        self.logger.info("Predict: beginning inference")
        self._model.eval()
        if not isinstance(x, (StackDataset, CombinedStackDataset, MoleculeDataset)):
            x = self.transform(x, is_training=False)

        self._datamodule_manager.set_predict_dataset(x)

        if batch_size is not None:
            self._datamodule_manager.set_batch_size(batch_size)
        if devices is None:
            devices = self._training_manager.params.devices
        if accelerator is None:
            accelerator = self._training_manager.params.accelerator

        trainer = L.Trainer(accelerator=accelerator, devices=devices, logger=False)
        preds = trainer.predict(self._model, datamodule=self.datamodule)
        preds = torch.cat(preds, axis=0)
        self.logger.info("Predict: inference finished")
        return preds

    def transform(
        self,
        mols: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Transforms a list of molecules into the input shape required by the model
        architecture. Will infer whether to fit standardizers depending on whether
        the y vector is passed or not.

        :param list[Mol] mols: list of rdkit molecules to featurize

        :param np.ndarray y: property labels for each molecule (if present). If not
            present, it will switch to test set mode

        :param list[str] | None bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None

        :param bool is_training: whether to fit a new standard scaler or not (
            relevant for validation sets for early stopping)

        :return StackDataset: featurized dataset ready to be batched for training
            or inference
        """
        return self._datamodule_manager.featurize(
            mols, y, is_training=is_training, bound_mask=bound_mask, n_jobs=n_jobs
        )

    def _inner_fit(self):
        """Inner logic for model training, delegated to the TrainingManager."""
        self._model = self._training_manager.run(
            model=self._model,
            datamodule=self.datamodule,
            architecture_cls=self._architecture,
            mlflow_manager=self._mlflow_manager,
            model_instance=self,
        )

    def fit(
        self,
        x: list[Mol] | StackDataset,
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        validation_set: StackDataset | None = None,
    ):
        """Runs training with the desired model architecture with early stopping on 10%
        of the training dataset. Optionally, will add a few more epochs on top with
        SWA.

        :param list[Mol] | StackDataset x: either a list of molecules, or a StackDataset computed by
            the appropriate datamodule, which contains both the input and labels
            for training

        :param np.ndarray | None y: either property labels in a numpy array or None
            if a StackDataset was passed for x, defaults to None

        :param list[str] | None bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None

        :param StackDataset | None validation_set: pre-transformed data to use for early-stopping,
            defaults to None
        """
        self.logger.info("Fit: beginning fit")
        self._datamodule_manager.prepare_fit_datasets(
            x=x,
            y=y,
            bound_mask=bound_mask,
            validation_set=validation_set,
            transform_fn=self.transform,
            early_stopping=self._training_manager.params.early_stopping,
            seed=self._training_manager.params.seed,
            batch_size=self._training_manager.params.batch_size,
        )

        self._inner_fit()

    def calibrate_uncertainty(
        self,
        calibration_mols: list[Mol],
        calibration_y: np.ndarray,
        num_iterations: int = 10,
        algorithm: str = "inductive_conformal",
        algorithm_args: dict | None = None,
    ):
        """Calibrate uncertainty estimates using a calibration set.

        :param list[Mol] calibration_mols: molecules for calibration
        :param np.ndarray calibration_y: true labels for calibration
        :param int num_iterations: MC dropout iterations
        :param str algorithm: calibration algorithm name
        :param dict | None algorithm_args: arguments for the calibration algorithm
        """
        if algorithm_args is None:
            algorithm_args = {"confidence_alpha": 0.2}
        self._uncertainty_manager.calibrate(
            model_instance=self,
            calibration_mols=calibration_mols,
            calibration_y=calibration_y,
            num_iterations=num_iterations,
            algorithm=algorithm,
            algorithm_args=algorithm_args,
        )

    def compute_uncertainty(
        self,
        x: list[Mol] | StackDataset,
        num_iterations: int = 10,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Computes uncertainty via Monte Carlo dropout for a test set.

        :param list[Mol] | StackDataset x: input to compute uncertainty for
        :param int num_iterations: how many iterations of dropout to do
        :param str | None accelerator: hardware to use for predictions
        :param int | None devices: how many resources to use
        :param int | None batch_size: batch size to use
        :return np.ndarray: array with the uncertainty for each prediction
        """
        return self._uncertainty_manager.compute(
            model_instance=self,
            x=x,
            num_iterations=num_iterations,
            accelerator=accelerator,
            devices=devices,
            batch_size=batch_size,
        )

    def compute_learned_embedding(
        self,
        x: list[Mol] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Returns learned fingerprints from the model. Can be either the output
        of the encoder or the representation from the last layer.

        :param list[Mol] | StackDataset x: input (X,) to compute learned embeddings for

        :param str | None accelerator: hardware to use, if None
            it is kept as training settings, defaults to None

        :param int | None devices: how many resources to use, if None
            it is kept as training settings, defaults to None

        :param int | None batch_size: batch size to use, if None
            it is kept as training settings, defaults to None

        :return np.ndarray: learned representation for the input, with dimensionality
            (X, fp_dim)
        """
        self.logger.info("Embedding extraction: beginning calculation")
        self._model.eval()
        if isinstance(x, list):
            x = self.transform(x, is_training=False)

        self._datamodule_manager.set_predict_dataset(x)

        if batch_size is not None:
            self._datamodule_manager.set_batch_size(batch_size)
        if devices is None:
            devices = self._training_manager.params.devices
        if accelerator is None:
            accelerator = self._training_manager.params.accelerator

        self._datamodule_manager.setup(stage="predict")
        encoding = self._model.compute_learned_embedding(
            self._datamodule_manager.predict_dataloader()
        )
        encoding = [enc.detach().numpy() for enc in encoding]
        self.logger.info("Embedding extraction: finished")
        return np.concatenate(encoding)

    def annotate(self, key: str, dictionary: dict):
        """Stores arbitrary dictionaries in params.metadata.extra under the specified key.
        If the key already exists, a warning is logged and the value is overwritten.

        :param str key: key to store the dictionary under in metadata.extra

        :param dictionary dict: dictionary to store in params.metadata.extra[key]
        """

        # Check if key already exists in metadata.extra and warn if overwriting
        if key in self._metadata.extra:
            self.logger.warning(
                f"Key '{key}' already exists in metadata.extra. Overwriting existing value."
            )
            self.logger.warning(f"Old value for '{key}': {self._metadata.extra[key]}")

        # Store the dictionary in metadata.extra
        self._metadata.extra[key] = dictionary

    def quantize(self):
        """Apply dynamic quantization to the model."""
        self._serialization_manager.quantize(self)

    def save_model(self, folder_path: str, quantize: bool = False):
        """Saves all components of a sklearn model into the target folder.

        :param str folder_path: folder where to store the components
        :param bool quantize: whether to quantize the model before saving
        """
        self._serialization_manager.save(self, folder_path, quantize)

    def get_input_args(self) -> dict:
        """Returns the input arguments for the model as a dictionary."""
        return self._serialization_manager.get_input_args(self)

    def export_to_yaml(self, path: str) -> None:
        """Export the model configuration YAMLs to a directory (no weights, no state).

        Writes the same ``config/`` subtree that :meth:`save_model` produces,
        so the output can be inspected, version-controlled, or used as a template.

        :param str path: directory where the config YAMLs will be written
        """
        self._serialization_manager.export_to_yaml(self, path)

    @classmethod
    def from_folder(cls, folder_path: str, accelerator: str = "cuda") -> object:
        """Constructor method to generate a fully restored copy of a saved sklearn model.

        :param str folder_path: location of folder with saved model artifacts
        :param str accelerator: accelerator to use for model loading
        :return object: fully restored class instance
        """
        return SerializationManager.load_from_folder(cls, folder_path, accelerator)

    @classmethod
    def from_config(cls, param_dict: dict) -> object:
        """Constructor method to create a new class instance "shell" from saved params.

        Expects a structured dict with top-level keys:
        ``model``, ``training``, ``datamodule``, ``metadata``,
        and optionally ``calibration``, ``tuning``.

        :param dict param_dict: structured parameter dictionary
        :return object: class instance with architecture but no fitted weights
        """
        return SerializationManager.load_from_config(cls, param_dict)

    def tune(
        self,
        train_set: list[StackDataset] | StackDataset,
        val_set: list[StackDataset] | StackDataset,
        architecture_search_budget: int = 30,
        architecture_grid: dict | None = None,
        optimizer_search_budget: int = 30,
        optimizer_grid: dict | None = None,
        scheduler_grid: dict | None = None,
    ) -> tuple:
        """Hyperparameter tuning routine.

        :param StackDataset train_set: dataset to use for training
        :param StackDataset val_set: target to tune parameters against
        :param int architecture_search_budget: iterations for architecture search
        :param dict | None architecture_grid: architecture parameters to tune
        :param int optimizer_search_budget: iterations for optimizer search
        :param dict | None optimizer_grid: optimizer parameters to tune
        :param dict | None scheduler_grid: scheduler parameters to tune
        :return tuple: study objects from Optuna for architecture and optimizer
        """
        return self._hpo_manager.tune(
            model_instance=self,
            train_set=train_set,
            val_set=val_set,
            architecture_search_budget=architecture_search_budget,
            architecture_grid=architecture_grid,
            optimizer_search_budget=optimizer_search_budget,
            optimizer_grid=optimizer_grid,
            scheduler_grid=scheduler_grid,
        )

    def explain_prediction(
        self,
        input: Mol,
        task_idx: int = 0,
        lime_bootstrap_num: int = 25,
        lime_descriptor_set: list[str] | None = None,
        use_std: bool = False,
    ):
        """Generate LIME explanations for a molecule prediction.

        :param Mol input: RDKit molecule to explain
        :param int task_idx: index of the task to explain
        :param int lime_bootstrap_num: number of bootstrap iterations for LIME
        :param list[str] | None lime_descriptor_set: RDKit descriptors for LIME
        :param bool use_std: use uncertainty estimates for predictions
        :return MatchaExplanation: LIME results with plotting methods
        """
        return self._explainability_manager.explain(
            model_instance=self,
            input_mol=input,
            task_idx=task_idx,
            lime_bootstrap_num=lime_bootstrap_num,
            lime_descriptor_set=lime_descriptor_set,
            use_std=use_std,
        )

    def configure_label_encoder(self, params: dict):
        """Configure the label encoder with the given parameters.

        :param dict params: label encoder configuration dictionary
        """
        self._datamodule_manager.configure_label_encoder(params)

    def configure_label_encoder_task(
        self,
        task_idx: int,
        task_label: str,
        class_thresholds: list[float] | None = None,
        class_labels: list[str] | None = None,
    ):
        """Configure a single task in the label encoder.

        :param int task_idx: index of the task to configure
        :param str task_label: human-readable label for the task
        :param list[float] | None class_thresholds: thresholds for binarization
        :param list[str] | None class_labels: class label names
        """
        self._datamodule_manager.configure_label_encoder_task(
            task_idx, task_label, class_thresholds, class_labels
        )

    def parse_output(
        self, output: np.ndarray, tag: str, convert_to_labels: bool = True
    ):
        """Parse raw model output into a structured format with label decoding.

        :param np.ndarray output: raw model predictions
        :param str tag: column name prefix for the output DataFrame
        :param bool convert_to_labels: whether to convert to class labels
        :returns: parsed output as a DataFrame
        """
        return self._datamodule_manager.parse_output(output, tag, convert_to_labels)

    def has_class_labels(self) -> bool:
        """Whether the label encoder has been configured with class labels.

        :returns: True if class labels are available
        :rtype: bool
        """
        return self._datamodule_manager.has_class_labels()


class ScikitLearnRegressorMixin:
    """Mixin to add regression-specific methods to scikit-learn models"""

    def predict(
        self,
        x: list[Mol] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Wraps self._inner_predict to account for regression
        scenarios, mimicking the functioning of the original sklearn method.

        :param list[Mol] | StackDataset x: input to compute predictions for

        :param str | None accelerator: hardware to use for predictions, if None
            it is kept as training settings, defaults to None

        :param int | None devices: how many resources to use, if None
            it is kept as training settings, defaults to None

        :param int | None batch_size: batch size to use, if None
            it is kept as training settings, defaults to None

        :return np.ndarray: predictions for the input, modulated depending whether
            the model is a classifier or regressor
        """
        preds = self._inner_predict(x, accelerator, devices, batch_size)
        preds = preds.numpy()
        preds = self._datamodule_manager.invert_y(preds)
        return preds

    def _adapt_dicts_for_mixin(
        self, datamodule_dict: dict, train_dict: dict
    ) -> tuple[dict, dict]:
        """Adapts datamodule and training dictionaries for regression.

        :param dict datamodule_dict: datamodule configuration dictionary
        :param dict train_dict: training configuration dictionary
        :return tuple[dict, dict]: adapted datamodule and training dictionaries
        """
        datamodule_dict["is_classification"] = False

        if "label_encoder_params" not in datamodule_dict:
            datamodule_dict["label_encoder_params"] = {}
        datamodule_dict["label_encoder_params"]["encoder_type"] = "regression"

        return datamodule_dict, train_dict

    def _default_predict(
        self,
        x: list[Mol] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Helper method to state which prediction routine to use
        for downstream applications (e.g. explainability)"""
        return self.predict(x, accelerator, devices, batch_size)


class ScikitLearnClassifierMixin:
    """Mixin to add classification-specific methods to scikit-learn models"""

    def predict_proba(
        self,
        x: list[Mol] | StackDataset,
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
        if isinstance(self._model, ChempropModel):
            return preds.numpy()
        else:
            transform = torch.nn.Sigmoid()
            preds = transform(preds).numpy()
            return preds

    def predict(
        self,
        x: list[Mol] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Wraps self._inner_predict to account for classification
        scenarios, mimicking the functioning of the original sklearn method.

        :param list[Mol] | StackDataset x: input to compute predictions for

        :param str | None accelerator: hardware to use for predictions, if None
            it is kept as training settings, defaults to None

        :param int | None devices: how many resources to use, if None
            it is kept as training settings, defaults to None

        :param int | None batch_size: batch size to use, if None
            it is kept as training settings, defaults to None

        :return np.ndarray: predictions for the input, modulated depending whether
            the model is a classifier or regressor
        """
        preds = self.predict_proba(x, accelerator, devices, batch_size)
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        return preds

    def _default_predict(
        self,
        x: list[Mol] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Helper method to state which prediction routine to use
        for downstream applications (e.g. explainability)"""
        return self.predict_proba(x, accelerator, devices, batch_size)

    def _adapt_dicts_for_mixin(
        self, datamodule_dict: dict, train_dict: dict
    ) -> tuple[dict, dict]:
        """Adapts datamodule and training dictionaries for classification.

        :param dict datamodule_dict: datamodule configuration dictionary
        :param dict train_dict: training configuration dictionary
        :return tuple[dict, dict]: adapted datamodule and training dictionaries
        """
        datamodule_dict["is_classification"] = True
        datamodule_dict["clip"] = False

        if "label_encoder_params" not in datamodule_dict:
            datamodule_dict["label_encoder_params"] = {}
        datamodule_dict["label_encoder_params"]["encoder_type"] = (
            "binary_classification"
        )

        return datamodule_dict, train_dict

    def encode_y(self, y: np.ndarray) -> np.ndarray:
        """Convert raw, continuous values into a one-hot encoded matrix
        for classification.

        :param np.ndarray y: continuous target values
        :return np.ndarray: one-hot encoded matrix
        """
        return self._datamodule_manager.encode_y(y)


ScikitLearnModelRegistry = ClassRegistry()
