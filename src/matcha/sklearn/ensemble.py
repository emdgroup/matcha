"""Ensemble wrapper for cross-validated scikit-learn-compatible models."""

from __future__ import annotations

import copy
import datetime
from typing import Any

import lightning as L
import numpy as np
from chemprop.data import MoleculeDataset
from rdkit.Chem.rdchem import Mol
from sklearn.model_selection import KFold
from torch.utils.data import StackDataset

from matcha import __version__
from matcha.datamodules.utils import CombinedStackDataset
from matcha.sklearn.base_sklearn_model import BaseScikitLearnModel
from matcha.sklearn.managers import (
    EnsembleMLFlowManager,
    EnsembleSerializationManager,
    EnsembleCalibrationManager,
)
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.sklearn_api import (
    MetadataInputModel,
    ScikitLearnEnsembleInputModel,
)


class Ensemble:
    """Ensemble of sklearn-API models trained via cross-validation.

    The ensemble is constructed from a *template* model instance
    (any concrete ``BaseScikitLearnModel`` subclass such as
    ``GINRegressor``).  The template is deep-copied *n_models* times,
    each copy receiving a different random seed.

    Example usage::

        from matcha.sklearn import GINRegressor, Ensemble

        model = GINRegressor(num_epochs=50, enc_num_layers=3)
        ensemble = Ensemble(model=model, n_models=5)
        ensemble.fit(train_mols, train_y)
        mean, std = ensemble.predict(test_mols)
    """

    def __init__(
        self,
        model: BaseScikitLearnModel,
        n_models: int = 5,
        seed: int = 0,
    ):
        """Create an ensemble from a template model instance.

        :param BaseScikitLearnModel model: a fully-constructed (but unfitted)
            sklearn-API model to use as a template.  Its class, architecture
            and hyper-parameters are inferred automatically.
        :param int n_models: how many CV members to create, defaults to 5
        :param int seed: base random seed (model *i* gets seed + *i*), defaults to 0
        """
        # basic attributes
        self._start_setup()

        # infer architecture name from the template
        self._architecture_name: str = model.__class__.__name__

        # create initial params config
        self.params = ScikitLearnEnsembleInputModel(
            architecture=self._architecture_name,
            learner=model.params,
            n_models=n_models,
            metadata=MetadataInputModel(**self._init_metadata()),
            calibration=None,
            mlflow=None,
            seed=seed,
        )

        # create ensemble members from the template
        self._create_model_box(model, n_models, seed)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def _create_model_box(
        self,
        template: BaseScikitLearnModel,
        n_models: int = 5,
        seed: int = 0,
    ):
        """Create *n_models* copies of *template*, each with a unique seed.

        Model 0 **is** the template itself; models 1 … n-1 are deep copies
        with their training seed adjusted.
        """
        self._model_box = []

        for i in range(n_models):
            self.logger.info(f"Creating model {i}")
            if i == 0:
                member = template
            else:
                member = copy.deepcopy(template)

            # set a unique seed for each member
            new_seed = seed + i
            member._training_manager.configure(
                {**member._training_manager.params.model_dump(), "seed": new_seed}
            )
            L.seed_everything(seed=new_seed, workers=True, verbose=False)
            self._model_box.append(member)

        # snapshot the learner params from the first model
        self.params.learner = self._model_box[0].params

    def _init_metadata(self):
        """Build the default metadata dictionary for a new ensemble instance.

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

    def _sync_params(self):
        """Push ensemble-level metadata to every member and refresh calibration."""
        for model in self._model_box:
            model._metadata = self.params.metadata
        if self.calibrator is not None:
            self.params.calibration = self._calibration_manager.params

    def _start_setup(self):
        """Initialize basic attributes and delegate managers for the ensemble."""
        self.logger = get_default_logger("ENSEMBLE")
        self._model_box = None
        self._cv_idx = None
        self._architecture_name: str | None = None

        # Delegate managers
        self._mlflow_manager = EnsembleMLFlowManager()
        self._serialization_manager = EnsembleSerializationManager()
        self._calibration_manager = EnsembleCalibrationManager()

        self.logger.info(f"Created  {self.__class__.__name__} instance")

    @property
    def calibrator(self):
        """The ensemble-level calibration object, or None."""
        return self._calibration_manager.calibrator

    @property
    def explainer(self):
        """The explainability object from the first ensemble member, or None."""
        return self._model_box[0].explainer if self._model_box else None

    @property
    def model_box(self):
        """List of individual model members in the ensemble."""
        return self._model_box

    @property
    def cv_idx(self):
        """Cross-validation split indices used during fitting, or None."""
        return self._cv_idx

    @property
    def is_classifier(self) -> bool:
        """Whether the ensemble members are classifiers."""
        return self._model_box[0].is_classifier

    @property
    def task_type(self) -> str:
        """Inferred from the first member: 'binary_classification' or 'regression'."""
        return self._model_box[0].task_type

    def _cv(
        self, stack_dataset: Any, bound_mask: list[str] | None, split_idx: int
    ) -> tuple[StackDataset]:
        """Helper method to split a featurized StackDataset into
        a train and test split, according to the cross-validation index.
        For example, when doing 5-fold CV with 5 models, model 0 will
        train on fold 0, model 1 on fold 1 and so forth.

        :param StackDataset stack: featurized dataset to split
        :param list[str] | None bound_mask: bound mask
        :param int split_idx: which split-model pair to use

        :return tuple[StackDataset]: train and test split to use for the
            i-th model
        """

        train_idx, val_idx = self._cv_idx[split_idx]

        if isinstance(stack_dataset, (StackDataset, CombinedStackDataset)):
            stack_train = stack_dataset.__getitems__(train_idx)
            stack_val = stack_dataset.__getitems__(val_idx)
            stack_train = {
                k: [x[k] for x in stack_train] for k in stack_train[0].keys()
            }
            stack_val = {k: [x[k] for x in stack_val] for k in stack_val[0].keys()}

            return StackDataset(**stack_train), StackDataset(**stack_val)

        elif isinstance(stack_dataset, MoleculeDataset):
            train = [stack_dataset.data[x] for x in train_idx]
            val = [stack_dataset.data[x] for x in val_idx]

            return MoleculeDataset(train), MoleculeDataset(val)

    def transform(
        self,
        x: list[Mol],
        y: np.ndarray = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
    ):
        """Transform molecules into the featurized dataset format.

        Delegates to the first ensemble member's transform. Fitting of
        scalers is deferred to the per-member CV loop inside :meth:`fit`.

        :param list[Mol] x: list of RDKit molecules to featurize
        :param np.ndarray y: property labels (optional)
        :param list[str] | None bound_mask: censor information on labels
        :param bool is_training: whether to fit standardizers (kept for API consistency)
        :returns: featurized dataset
        :rtype: StackDataset
        """
        return self._model_box[0].transform(x, y, bound_mask, is_training=is_training)

    def fit(
        self,
        x: list[Mol] | StackDataset,
        y: np.ndarray,
        bound_mask: list[str] | None = None,
    ):
        """Fits each model in the box on the training data. Each model
        will be fit on a different CV split.

        :param list[Mol] | tuple x: training data, as for normal sklearn API models

        :param np.ndarray | None y: labels to predict

        :param list[str] | None bound_mask: censor information on the labels
        """

        # this does not use the self.transform method, but rather forces fitting, then
        # splits. This has the disadvantage that early stopping won't be completely
        # accurate. TODO: fix this, see above in self._cv
        if isinstance(x, list):
            dataset = self._model_box[0].transform(x, y, bound_mask, is_training=True)
        else:
            dataset = x

        kf = KFold(
            n_splits=self.params.n_models, shuffle=True, random_state=self.params.seed
        )
        cv_idx = [x for x in kf.split(x)]

        if len(x) == len(dataset):
            self._cv_idx = cv_idx
        else:
            factor = len(dataset) // len(x)
            self._cv_idx = [
                (np.repeat(train_idx, factor), np.repeat(test_idx, factor))
                for train_idx, test_idx in cv_idx
            ]

        for i in range(1, len(self._model_box)):
            self._model_box[i]._datamodule_manager.datamodule = copy.deepcopy(
                self._model_box[0].datamodule
            )

        # read early_stopping from the first model's training manager
        early_stopping = self._model_box[0]._training_manager.params.early_stopping

        for i, model in enumerate(self._model_box):
            if early_stopping:
                train, val = self._cv(dataset, bound_mask, i)
            else:
                train = dataset
                val = None

            if self._mlflow_manager.is_active:
                self._mlflow_manager.setup_member_mlflow(model, i)
            self.logger.info(f"Beginning fit of model {i}")
            model.fit(x=train, validation_set=val)
            self.logger.info(f"Finished fit of model {i}")

        # update post-fit params (e.g. dictionary for CLMs)
        self.params.learner = self.model_box[0].params

        if self._mlflow_manager.is_active:
            self._mlflow_manager.log_ensemble_params(self.params.model_dump())

    def predict(
        self,
        x: list[Mol] | tuple,
        reduce: bool = True,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray]:
        """Predicts a test sample with all models in the ensemble.
        Optionally returns the unreduced numpy array.

        :param list[Mol] | tuple x: molecules to predict

        :param bool reduce: whether to compute mean and std from ensemble,
            defaults to True

        :param str | None accelerator: hardware to use for predictions, if None
            it is kept as training settings, defaults to None

        :param int | None devices: how many resources to use, if None
            it is kept as training settings, defaults to None

        :param int | None batch_size: batch size to use, if None
            it is kept as training settings, defaults to None

        :return np.ndarray | tuple[np.ndarray]: predictions from the ensemble
        """
        x = self._model_box[0].transform(x, is_training=False)

        pred_box = [
            model._default_predict(x, accelerator, devices, batch_size)
            for model in self._model_box
        ]

        pred_box = np.stack(pred_box, axis=2)

        if reduce is False:
            if pred_box.shape[1] == 1:
                return pred_box[:, 0, :]
            return pred_box
        else:
            means = np.mean(pred_box, axis=2)
            std = np.std(pred_box, axis=2)
            std = self._calibration_manager.compute_uncertainty(std)
            return means, std

    def save_model(self, path: str, quantize: bool = False):
        """Saves each member of the ensemble into the target folder.

        :param str path: folder where to store the components
        :param bool quantize: whether to quantize models before saving
        """
        self._serialization_manager.save(self, path, quantize)

    def export_to_yaml(self, path: str) -> None:
        """Exports the model configuration and parameters to a YAML file.

        This method uses the export_to_yaml method from the first model in
        the ensemble, then adds the ensemble param.
        """
        self._serialization_manager.export_to_yaml(self, path)

    @classmethod
    def from_config(cls, params: dict):
        """Reconstruct an Ensemble from a saved parameter dictionary.

        This creates individual model instances using the registered architecture's
        ``from_config``, then wraps the first one into a new Ensemble via the
        standard ``__init__``.

        :param dict params: saved ensemble parameters (as produced by
            ``self.params.model_dump()``)
        :return Ensemble: reconstructed ensemble instance
        """
        return EnsembleSerializationManager.load_from_config(cls, params)

    @classmethod
    def from_folder(cls, path: str, accelerator: str = "cuda"):
        """Load a saved Ensemble from a folder.

        Each member is individually loaded via its architecture's
        ``from_folder``, then assembled into an Ensemble shell.

        :param str path: folder containing ensemble artifacts
        :param str accelerator: device to load models onto
        :return Ensemble: fully restored ensemble
        """
        return EnsembleSerializationManager.load_from_folder(cls, path, accelerator)

    def set_mlflow_experiment(
        self,
        experiment_name: str,
        run_name: str = "model",
        tag: dict | None = None,
        log_dir: str | None = "./matcha_log",
        server_uri: str | None = None,
    ):
        """Sets the experiment name and run name for the model, useful for logging purposes

        :param str experiment_name: name of the experiment to set
        :param str run_name: name of the run to associate with the experiment
        :param dict | None tag: optional tags to associate with the run
        :param str | None log_dir: optional directory for logging
        :param str | None server_uri: optional MLflow server URI
        """
        self.params.mlflow = self._mlflow_manager.setup_experiment(
            experiment_name=experiment_name,
            run_name=run_name,
            tag=tag,
            log_dir=log_dir,
            server_uri=server_uri,
        )

    def custom_mlflow_log(self, artifact_path: str, tag: dict) -> None:
        """Log an artifact to MLflow via the manager.

        :param str artifact_path: path to the artifact file to log
        :param dict tag: tags specific to this artifact
        """
        self._mlflow_manager.log_artifact(artifact_path, tag)

    def calibrate_uncertainty(
        self,
        calibration_mols: list[Mol],
        calibration_y: np.ndarray,
        algorithm: str = "inductive_conformal",
        algorithm_args: dict | None = None,
    ):
        """Calibrate uncertainty estimates using a calibration set.

        :param list[Mol] calibration_mols: molecules for calibration
        :param np.ndarray calibration_y: true labels for calibration
        :param str algorithm: calibration algorithm name
        :param dict | None algorithm_args: arguments for the calibration algorithm
        """
        self._calibration_manager.calibrate(
            ensemble=self,
            calibration_mols=calibration_mols,
            calibration_y=calibration_y,
            algorithm=algorithm,
            algorithm_args=algorithm_args,
        )

    def annotate(self, key: str, dictionary: dict):
        """Stores arbitrary dictionaries in params.metadata.extra under the specified key.
        If the key already exists, a warning is logged and the value is overwritten.

        :param str key: key to store the dictionary under in metadata.extra

        :param dictionary dict: dictionary to store in params.metadata.extra[key]
        """

        # Store the dictionary in metadata.extra
        self.params.metadata.extra[key] = dictionary
        self._sync_params()

    def configure_label_encoder(self, params: dict):
        """Configure the label encoder for all ensemble members.

        :param dict params: label encoder configuration dictionary
        """
        for model in self._model_box:
            model.configure_label_encoder(params)

    def configure_label_encoder_task(
        self,
        task_idx: int,
        task_label: str,
        class_thresholds: list[float] | None = None,
        class_labels: list[str] | None = None,
    ):
        """Configure a single task in the label encoder for all members.

        :param int task_idx: index of the task to configure
        :param str task_label: human-readable label for the task
        :param list[float] | None class_thresholds: thresholds for binarization
        :param list[str] | None class_labels: class label names
        """
        for model in self._model_box:
            model.configure_label_encoder_task(
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
        return self._model_box[0].parse_output(output, tag, convert_to_labels)

    def has_class_labels(self) -> bool:
        """Whether the label encoder has been configured with class labels.

        :returns: True if class labels are available
        :rtype: bool
        """
        return self._model_box[0].has_class_labels()

    def encode_y(self, y: np.ndarray) -> np.ndarray:
        """Function to convert raw, continuous values into a one-hot
        encoded matrix for classification.

        Works only for classifiers, and assumes that the label encoder is the same
        for every sub-model.
        """
        return self.model_box[0]._datamodule_manager.encode_y(y)
