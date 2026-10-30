import numpy as np
from collections.abc import Iterable

from chemprop.data import MoleculeDataset
from rdkit.Chem.rdchem import Mol
from torch.utils.data import StackDataset

from matcha.datamodules.utils import CombinedStackDataset
from matcha.sklearn.utils import random_split
from matcha.utils.logging import get_default_logger


class DataModuleManager:
    """Manages the Lightning DataModule lifecycle: creation, featurization,
    dataset assignment, and label encoding.

    The underlying datamodule object is created by the concrete sklearn model
    subclass via ``create_from_factory``, and from that point on the manager
    owns the reference and exposes a controlled interface.
    """

    def __init__(self):
        self._datamodule = None
        self.logger = get_default_logger("DATAMODULE")

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    @property
    def datamodule(self):
        """The underlying Lightning DataModule, or None if not yet created."""
        return self._datamodule

    @datamodule.setter
    def datamodule(self, value):
        """Allow direct assignment for factory / deserialization flows."""
        self._datamodule = value

    @property
    def params(self):
        """Params owned by the underlying datamodule, or None."""
        if self._datamodule is not None:
            return self._datamodule.params
        return None

    # ------------------------------------------------------------------
    # Featurization
    # ------------------------------------------------------------------

    def featurize(
        self,
        mols: list[Mol],
        y: np.ndarray | None = None,
        is_training: bool = True,
        bound_mask: list[str] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Featurize molecules via the underlying datamodule.

        :param list[Mol] mols: list of rdkit molecules to featurize
        :param np.ndarray | None y: property labels (None → inference mode)
        :param bool is_training: whether to fit scalers
        :param list[str] | None bound_mask: censoring mask
        :param int | None n_jobs: parallelism for featurization
        :return StackDataset: featurized dataset
        """
        self.logger.info("Transform: generating features")
        if y is None:
            dataset = self._datamodule.featurize(
                mols, None, is_training=False, bound_mask=bound_mask, n_jobs=n_jobs
            )
        else:
            dataset = self._datamodule.featurize(
                mols, y, is_training=is_training, bound_mask=bound_mask, n_jobs=n_jobs
            )
        self.logger.info("Transform: features generated")
        return dataset

    # ------------------------------------------------------------------
    # Dataset assignment helpers
    # ------------------------------------------------------------------

    def set_train_dataset(self, dataset):
        """Assign the training dataset on the underlying datamodule."""
        self._datamodule.dataset_train = dataset

    def set_val_dataset(self, dataset):
        """Assign the validation dataset on the underlying datamodule."""
        self._datamodule.dataset_val = dataset

    def set_predict_dataset(self, dataset):
        """Assign the prediction dataset on the underlying datamodule."""
        self._datamodule.dataset_predict = dataset

    def set_batch_size(self, batch_size: int):
        """Update the batch size on the underlying datamodule."""
        self._datamodule.params.batch_size = batch_size

    # ------------------------------------------------------------------
    # Label encoding delegation
    # ------------------------------------------------------------------

    def configure_label_encoder(self, params: dict):
        """Configure the label encoder on the underlying datamodule.

        :param dict params: label encoder configuration
        """
        params["encoder_type"] = (
            self._datamodule.params.label_encoder_params.encoder_type
        )
        self._datamodule.configure_label_encoder(params)

    def configure_label_encoder_task(
        self,
        task_idx: int,
        task_label: str,
        class_thresholds: list[float] | None = None,
        class_labels: list[str] | None = None,
    ):
        """Configure a single task in the label encoder.

        :param int task_idx: index of the task
        :param str task_label: label for the task
        :param list[float] | None class_thresholds: thresholds for classification
        :param list[str] | None class_labels: class label names
        """
        self._datamodule.configure_label_encoder_task(
            task_idx, task_label, class_thresholds, class_labels
        )

    def parse_output(
        self, output: np.ndarray, tag: str, convert_to_labels: bool = True
    ):
        """Parse model output through the datamodule's label decoder.

        :param np.ndarray output: raw model output
        :param str tag: output tag
        :param bool convert_to_labels: whether to convert to labels
        :return: parsed output
        """
        return self._datamodule.parse_output(output, tag, convert_to_labels)

    def has_class_labels(self) -> bool:
        """Whether the datamodule's label encoder has class labels configured.

        :return bool: True if class labels are configured
        """
        return self._datamodule.has_class_labels()

    def invert_y(self, preds: np.ndarray) -> np.ndarray:
        """Invert the y transform on predictions (e.g. unscale).

        :param np.ndarray preds: model predictions
        :return np.ndarray: un-transformed predictions
        """
        return self._datamodule.invert_y(preds)

    def state_dict(self) -> dict:
        """Return the datamodule state dict for serialization.

        :return dict: state dict
        """
        return self._datamodule.state_dict()

    def load_state_dict(self, state_dict: dict):
        """Load datamodule state from a state dict.

        :param dict state_dict: state dict to load
        """
        self._datamodule.load_state_dict(state_dict)

    def setup(self, stage: str):
        """Proxy for the underlying datamodule's setup method.

        :param str stage: one of 'fit', 'test', 'predict'
        """
        self._datamodule.setup(stage=stage)

    def predict_dataloader(self):
        """Proxy for the underlying datamodule's predict_dataloader method."""
        return self._datamodule.predict_dataloader()

    # ------------------------------------------------------------------
    # Fit dataset preparation
    # ------------------------------------------------------------------

    def prepare_fit_datasets(
        self,
        x,
        y: np.ndarray | None,
        bound_mask: list[str] | None,
        validation_set: StackDataset | None,
        transform_fn,
        early_stopping: bool,
        seed: int,
        batch_size: int,
    ):
        """Prepare training (and optionally validation) datasets for fit.

        Handles three input modes:
        1. Pre-featurized StackDataset / CombinedStackDataset / MoleculeDataset
        2. Raw molecule list with early stopping (auto-splits 90/10)
        3. Raw molecule list without early stopping

        :param x: molecules or pre-featurized dataset
        :param np.ndarray | None y: labels
        :param list[str] | None bound_mask: censoring mask
        :param StackDataset | None validation_set: optional pre-featurized val set
        :param transform_fn: callable to featurize molecules (model.transform)
        :param bool early_stopping: whether to create a validation split
        :param int seed: random seed for splitting
        :param int batch_size: batch size to set on the datamodule
        """
        if isinstance(x, (CombinedStackDataset, StackDataset, MoleculeDataset)):
            self.set_train_dataset(x)
            if validation_set is not None:
                self.set_val_dataset(validation_set)

        elif isinstance(x, Iterable) and (
            isinstance(x[0], Mol) or isinstance(x[0], str)
        ):
            if early_stopping:
                x_train, y_train, bound_train, x_val, y_val, bound_val = random_split(
                    x, y, bound_mask, seed=seed
                )
                self.set_train_dataset(
                    transform_fn(
                        mols=x_train,
                        y=y_train,
                        bound_mask=bound_train,
                        is_training=True,
                        n_jobs=None,
                    )
                )
                self.set_val_dataset(
                    transform_fn(
                        mols=x_val,
                        y=y_val,
                        bound_mask=bound_val,
                        is_training=False,
                        n_jobs=1,
                    )
                )
            else:
                self.set_train_dataset(
                    transform_fn(
                        mols=x,
                        y=y,
                        bound_mask=bound_mask,
                        is_training=True,
                        n_jobs=None,
                    )
                )
        else:
            raise ValueError(
                f"x must be either a list of molecules or a StackDataset, but got {type(x)}"
            )

        self.set_batch_size(batch_size)

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------

    def encode_y(self, y: np.ndarray) -> np.ndarray:
        """Convert raw, continuous values into a one-hot encoded matrix
        for classification, using the datamodule's label encoder.

        :param np.ndarray y: continuous target values
        :return np.ndarray: one-hot encoded matrix
        """
        return self._datamodule.encode_y(y)
