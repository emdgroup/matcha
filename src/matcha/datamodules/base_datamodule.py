"""Base data module providing shared featurization, scaling, and label encoding logic."""

from abc import ABC, abstractmethod
import os
import numpy as np
from chemprop.data import MoleculeDataset
from lightning import LightningDataModule
from rdkit.Chem.rdchem import Mol
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from torch import tensor, float32
from torch.utils.data import DataLoader, StackDataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from matcha.datamodules.classic.label_encoder import (
    LabelEncoderRegistry,
    RegressionLabelEncoder,
    BinaryClassificationLabelEncoder,
)
from matcha.datamodules.classic.label_transform import LabelTransform
from matcha.datamodules.utils import CombinedStackDataset
from matcha.utils.registry import ClassRegistry
from matcha.utils.schemas import MolDataset

from rdkit import Chem
from rdkit.Chem import ResonanceMolSupplier, ResonanceFlags


class BaseDataModule(ABC, LightningDataModule):
    """Base class for all datamodules.

    It is not meant to be instantiated directly, but rather to be used as a parent
    class for each datamodule.
    The most important method for the children classes is :method:`self.featurize`. Please
    check out the docs for that method for more information.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        augment_resonance: bool = False,
        label_encoder_params: dict = {},
        label_transform_params: dict = {},
    ):
        """Initialize the base data module.

        :param str scaler_type: type of y scaler to use (``'standard'`` or ``'quantile'``)
        :param bool augment_resonance: whether to augment training data with resonance structures
        :param dict label_encoder_params: parameters for the label encoder
        :param dict label_transform_params: parameters for the label transform
        """
        super().__init__()

        self.collate_fn_map = {}
        self._dataset_train = None
        self._dataset_predict = None
        self._dataset_val = None
        self._dataset_test = None
        self._dataloader_train = None
        self._dataloader_val = None
        self._dataloader_test = None
        self._dataloader_predict = None
        self._augment_resonance = augment_resonance
        self._create_y_scaler(scaler_type)
        self._create_label_encoder(label_encoder_params)
        self._create_label_transform(label_transform_params)

    def _set_dataset(self, attr_name: str, dataset: StackDataset):
        if isinstance(dataset, (StackDataset, CombinedStackDataset, MoleculeDataset)):
            setattr(self, f"_{attr_name}", dataset)
        else:
            raise ValueError(f"Cannot store {attr_name} as dataset")

    def _get_dataset(self, attr_name: str) -> StackDataset:
        return getattr(self, f"_{attr_name}")

    def _create_y_scaler(
        self, scaler_type: str
    ) -> StandardScaler | QuantileTransformer:
        """Factory method to create y scaler instances.

        :param str scaler_type: type of scaler to create ('standard' or 'quantile')
        :return: configured scaler instance
        :raises ValueError: if scaler_type is not 'standard' or 'quantile'
        """
        if scaler_type == "standard":
            self._y_scaler = StandardScaler()
            self.params.scaler_type = scaler_type
        elif scaler_type == "quantile":
            self._y_scaler = QuantileTransformer(
                output_distribution="uniform", subsample=None
            )
            self.params.scaler_type = scaler_type
        else:
            raise ValueError(
                f"Unknown scaler_type '{scaler_type}'. Must be 'standard' or 'quantile'."
            )

    def _create_label_encoder(
        self, label_encoder_params: dict
    ) -> RegressionLabelEncoder | BinaryClassificationLabelEncoder:
        """Factory method to create label encoder instances.

        :param dict label_encoder_params: parameters for label encoder configuration
        :return: configured label encoder instance
        """
        if label_encoder_params != {}:
            # If encoder_type is specified, use it; otherwise default to regression
            encoder_type = label_encoder_params.pop("encoder_type", "regression")
            self._label_encoder = LabelEncoderRegistry[encoder_type](
                label_encoder_params
            )
        else:
            self._label_encoder = LabelEncoderRegistry["regression"]()
        self.params.label_encoder_params = self._label_encoder.params

    def _create_label_transform(
        self, label_transform_params: dict | None
    ) -> LabelTransform:
        """Factory method to create label transform instances.

        :param label_transform_map: transformation mapping configuration
        :return: configured label transform instance
        """
        if label_transform_params == {} or label_transform_params is None:
            self._label_transform = LabelTransform()
        else:
            self._label_transform = LabelTransform(**label_transform_params)
        self.params.label_transform_params = self._label_transform.params
        return self._label_transform

    @property
    def dataset_train(self) -> StackDataset | None:
        return self._get_dataset("dataset_train")

    @dataset_train.setter
    def dataset_train(self, dataset: StackDataset):
        self._set_dataset("dataset_train", dataset)

    @property
    def dataset_val(self) -> StackDataset | None:
        return self._get_dataset("dataset_val")

    @dataset_val.setter
    def dataset_val(self, dataset: StackDataset):
        self._set_dataset("dataset_val", dataset)

    @property
    def dataset_test(self) -> StackDataset | None:
        return self._get_dataset("dataset_test")

    @dataset_test.setter
    def dataset_test(self, dataset: StackDataset):
        self._set_dataset("dataset_test", dataset)

    @property
    def dataset_predict(self) -> StackDataset | None:
        return self._get_dataset("dataset_predict")

    @dataset_predict.setter
    def dataset_predict(self, dataset: StackDataset):
        self._set_dataset("dataset_predict", dataset)

    @property
    def y_scaler(self) -> StandardScaler | QuantileTransformer:
        """Property to access the y scaler instance."""
        return self._y_scaler

    @property
    def label_encoder(
        self,
    ) -> RegressionLabelEncoder | BinaryClassificationLabelEncoder:
        """Property to access the label encoder instance."""
        return self._label_encoder

    @property
    def label_transform(self) -> LabelTransform:
        """Property to access the label transform instance."""
        return self._label_transform

    def train_dataloader(self):
        return self._dataloader_train

    # this is crazy, but apparently Lightning always requires a val dataloader
    # to be defined inside a datamodule, even if you have no early stopping.
    # because of this, we need to make a dummy one if dataset_val is None
    def val_dataloader(self):
        if isinstance(self._dataloader_val, DataLoader):
            return self._dataloader_val
        else:
            return DataLoader({})

    def test_dataloader(self):
        return self._dataloader_test

    def predict_dataloader(self):
        return self._dataloader_predict

    def _validate_datamodule_input(self, x, y, bound_mask):
        """Simple validation function to check that the input dataset
        has adequate types. In case it doesn't, either throw an error or fix it."""
        valid = MolDataset(mols=x, y=y, bound_mask=bound_mask)
        return valid.mols, valid.y, valid.bound_mask

    def _validate_input(
        self,
        x: list[Mol] | list[str],
        y: np.ndarray | None,
        bound_mask: list[str] | None,
        n_jobs: int | None,
    ):
        """Validates inputs and handles basic preprocessing without scaling.

        :param x: list of molecules or SMILES strings
        :param y: labels array or None
        :param bound_mask: bound mask or None
        :param n_jobs: number of jobs or None
        :return: validated inputs
        """
        # validate input
        x, y, bound_mask = self._validate_datamodule_input(x, y, bound_mask)

        # parse y - handle empty case but don't scale
        if isinstance(y, type(None)):
            y = self._handle_empty_y(len(x))

        # if n_jobs is None, guess good n_jobs value
        if n_jobs is None:
            n_jobs = self._guess_n_jobs(x)

        return x, y, bound_mask, n_jobs

    @abstractmethod
    def featurize(
        self,
        x: list[Mol],
        y: np.ndarray | None,
        is_training: bool = True,
        bound_mask: list[str] | None = None,
    ) -> StackDataset:
        """Converts a list of rdkit molecules into a format fit for neural network training,
        depending on the architecture to be used downstream.

        This method generates features and applies scaling as needed. It toggles between
        fitting (when is_training=True) and transforming (when is_training=False) scalers.

        Returns a StackDataset in dictionary format, with the following keys:
        - 'y': a tensor containing the labels to predict (required)
        - 'mol_features': a tensor containing fixed-size molecular features
        - 'graph': a DGLGraph object containing the molecular graph, e.g. for GNNs
        - 'token_ids': a tensor containing the tokenized SMILES strings, e.g. for CLMs

        Depending on the architecture, more inputs might be required.

        :param list[Mol] x: list of rdkit molecules to featurize

        :param np.ndarray | None y: labels to predict for each molecule, or None

        :param bool is_training: Whether to fit the X and Y scalers on the input,
            or leverage pre-existing ones to only normalize

        :param list[str] | None bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None

        :return StackDataset: output ready to be batched by a torch.Dataloader
        """

    @abstractmethod
    def state_dict(self) -> dict:
        """Returns the parameters necessary for datamodule tracking. The
        parameters should be outputted inside a dictionary.

        Necessary for MLFlow monitoring.
        """

    def _guess_n_jobs(self, x: list[Mol]) -> int:
        """Set number of jobs depending on the number of samples X.

        The number of jobs is set to either len(x)/1000, 1 or 64 depending on the case.

        :param list[Mol] x: list of rdkit molecules that need to be featurized

        :return int: number of cores to use when generating features
        """
        n_jobs = int(len(x) / 1000)
        if n_jobs < 1:
            n_jobs = 1
        elif n_jobs > min(os.cpu_count() // 2, 32):
            n_jobs = min(os.cpu_count() // 2, 32)
        return n_jobs

    def _handle_empty_y(self, target_size: int) -> np.ndarray:
        """Handles the case where an empty y vector is passed

        Creates a dummy array with the correct shape. It is meant to be used with test datasets.

        :param int taget_size: number of compounds to process, needed to generate
            a matching dummy y array filled with '404' to indicate missing values

        :return np.ndarray: dummy array with appropriate shape
        """
        try:
            n_tasks = self._y_scaler.n_features_in_
        except Exception:
            n_tasks = 1
        return np.zeros((target_size, n_tasks)) + 404

    def _fit_y(self, dataset: StackDataset) -> None:
        """Utility to update y scaler, modifying dataset in-place.

        :param dataset: StackDataset to fit and transform
        """
        if not self.params.is_classification:
            y = dataset.datasets["y"].numpy()
            y = np.copy(y)

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            if self.params.clip is True:
                y_clip = {"Min": np.nanmin(y), "Max": np.nanmax(y)}
                y_clip = {k: float(v) for k, v in y_clip.items()}
                self._label_transform.set_clipping_bounds(y_clip)
                self.params.label_transform_params = self._label_transform.params
            y_scaled = self._label_transform.process(y, forward=True)

            self._y_scaler.fit(y_scaled)
        else:
            pass

    def _transform_y(self, dataset: StackDataset) -> None:
        """Utility to transform y using scaler, modifying dataset in-place.

        :param dataset: StackDataset to transform
        """
        if not self.params.is_classification:
            y = dataset.datasets["y"].numpy()
            y = np.copy(y)

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            y = self._label_transform.process(y, forward=True)
            if hasattr(self._y_scaler, "n_features_in_"):
                y_transformed = self._y_scaler.transform(y)
                dataset.datasets["y"] = tensor(y_transformed, dtype=float32)
            else:
                dataset.datasets["y"] = tensor(y, dtype=float32)
        else:
            pass

    def _invert_y(self, dataset: StackDataset) -> None:
        """Utility to transform y using scaler, modifying dataset in-place.

        :param dataset: StackDataset to transform
        """
        y = dataset.datasets["y"].numpy()
        y = np.copy(y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if hasattr(self._y_scaler, "n_features_in_"):
            y = self._y_scaler.inverse_transform(y)
        else:
            print("No scaler found for Y, skipping...")

        y = self._label_transform.process(y, forward=False)
        dataset.datasets["y"] = tensor(y, dtype=float32)

    def fit(self, dataset: StackDataset) -> None:
        """Fits scalers and other stateful transformations on the dataset.

        :param StackDataset dataset: dataset to fit transformations on
        """
        self._fit_y(dataset)

    def transform(self, dataset: StackDataset) -> StackDataset:
        """Applies fitted transformations to the dataset.

        :param StackDataset dataset: dataset to transform
        :return StackDataset: transformed dataset (modified in-place)
        """
        self._transform_y(dataset)

        return dataset

    def _augment_single(self, args):
        """
        Helper for augment: performs augmentation for a single molecule.
        args: tuple (mol, y_i, bound_i, use_resonance, max_n)
        Returns: (aug_mols, aug_y, aug_bound)
        """
        mol, y_i, bound_i, use_resonance, max_n = args
        mols_aug = []
        y_aug = []
        bound_aug = []
        if use_resonance:
            resonance = ResonanceMolSupplier(
                mol, flags=ResonanceFlags.ALLOW_CHARGE_SEPARATION
            )
            resonance = [x for x in resonance]
            if len(resonance) > max_n:
                resonance = resonance[:max_n]
            try:
                resonance = [Chem.MolFromSmiles(Chem.MolToSmiles(m)) for m in resonance]
                resonance = [x for x in resonance if x is not None]
                mols_aug.extend(resonance)
                y_aug.extend([y_i] * len(resonance))
                if bound_i is not None:
                    bound_aug.extend([bound_i] * len(resonance))
            except Exception:
                pass
        return mols_aug, y_aug, bound_aug

    def augment(
        self,
        mol_list: list[Mol],
        y: np.ndarray,
        bound_mask: list[str] | None = None,
        use_resonance: bool = True,
        max_n: int = 5,
        n_jobs: int = 1,
    ) -> tuple[list[Mol], np.ndarray, list[str] | None]:
        """Augment the dataset with resonance structures.

        :param list[Mol] mol_list: list of rdkit molecules to augment
        :param np.ndarray y: labels array of shape (N, T)
        :param list[str] | None bound_mask: bound mask or None
        :param bool use_resonance: whether to use resonance augmentation
        :param int max_n: maximum number of resonance structures per molecule
        :param int n_jobs: number of parallel jobs
        :returns: tuple of (augmented molecules, augmented labels, augmented bound mask)
        :rtype: tuple[list[Mol], np.ndarray, list[str] | None]
        """
        # validate inputs without scaling
        mol_list, y, bound_mask, n_jobs = self._validate_input(
            mol_list, y, bound_mask, n_jobs
        )

        # Prepare arguments for each molecule
        args_list = []
        for i, mol in enumerate(mol_list):
            y_i = y[i]
            bound_i = bound_mask[i] if bound_mask is not None else None
            args_list.append((mol, y_i, bound_i, use_resonance, max_n))

        if n_jobs > 1:
            from matcha.utils.wrapper import parallelize

            results = parallelize(
                lambda batch: [self._augment_single(arg) for arg in batch],
                args_list,
                n_jobs,
            )
        else:
            results = [self._augment_single(arg) for arg in args_list]

        mols_augmented = []
        y_augmented = []
        bound_augmented = []
        for mols, ys, bounds in results:
            mols_augmented.extend(mols)
            y_augmented.extend(ys)
            bound_augmented.extend(bounds) if bounds else None

        return (
            mols_augmented,
            np.array(y_augmented),
            bound_augmented if bound_augmented else None,
        )

    def _process_y(
        self,
        dataset: StackDataset,
        bound_mask: list[str] | list[list[str]] | None = None,
    ) -> None:
        """
        Modifies the dataset in-place to account for censor information
        or classification settings.

        :param dataset: StackDataset with labels to scale
        :param bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None
        """
        y = dataset.datasets["y"].numpy()
        y = np.copy(y)

        if not self.params.is_classification:
            if bound_mask is not None:
                mask = np.zeros((y.shape[0], y.shape[1]))
                if isinstance(bound_mask[0], list):
                    for i, sublist in enumerate(bound_mask):
                        sublist = [s if isinstance(s, str) else "nan" for s in sublist]
                        lt = [j for j, s in enumerate(sublist) if "<" in s]
                        gt = [j for j, s in enumerate(sublist) if ">" in s]
                        mask[lt, i] = -1
                        mask[gt, i] = 1
                else:
                    lt = [i for i, s in enumerate(bound_mask) if "<" in s]
                    gt = [i for i, s in enumerate(bound_mask) if ">" in s]
                    mask[lt] = -1
                    mask[gt] = 1
                y = np.stack((y, mask), axis=2)
                # Update the dataset with the bound mask modifications
                dataset.datasets["y"] = tensor(y, dtype=float32)
        elif self.params.is_classification and self._label_encoder.is_set():
            stack = []
            for i in range(y.shape[1]):
                ith = self._label_encoder._continuous_to_categorical(y[:, i], i)
                stack.append(ith)
            y = np.concatenate(stack, axis=1)
            # Update the dataset with the classification modifications
            dataset.datasets["y"] = tensor(y, dtype=float32)

    def encode_y(self, y: np.ndarray) -> np.ndarray:
        """Convert raw, continuous values into a one-hot encoded matrix
        for classification using the label encoder.

        :param np.ndarray y: continuous target values of shape (N, T)
        :return np.ndarray: one-hot encoded matrix
        """
        return self._label_encoder._all_to_categorical(y)

    def invert_y(self, y: np.array) -> None:
        """Inverts all y transformations from numpy arrays.

        :param y: np.array to invert
        """
        y = np.copy(y)
        y = self._y_scaler.inverse_transform(y)
        y = self._label_transform.process(y, forward=False)
        return y

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate a batch of samples into a single dictionary of batched tensors.

        :param list[dict] batch: list of sample dictionaries from the dataset
        :returns: dictionary with collated tensors for each key
        :rtype: dict
        """

        def _identity(x, collate_fn_map=None):
            return x

        collate_fn_map = default_collate_fn_map.copy()
        if self.collate_fn_map is not None:
            collate_fn_map.update(self.collate_fn_map)
        elem = batch[0]
        for value in elem.values():
            if type(value) not in collate_fn_map:
                collate_fn_map[type(value)] = _identity
        return collate(batch, collate_fn_map=collate_fn_map)

    def _create_dataloader(
        self, dataset: StackDataset, is_training: bool
    ) -> DataLoader:
        # Check if the last batch would have only 1 sample
        # This prevents BatchNorm errors during training
        dataset_size = len(dataset)
        batch_size = self.params.batch_size
        last_batch_size = dataset_size % batch_size
        drop_last = (last_batch_size == 1) and is_training

        return DataLoader(
            dataset=dataset,
            batch_size=self.params.batch_size,
            shuffle=is_training,
            collate_fn=self.collate_fn,
            num_workers=self.params.num_workers,
            drop_last=drop_last,
        )

    def setup(self, stage: str):
        """Set up dataloaders for a given Lightning stage.

        :param str stage: one of ``'fit'``, ``'test'``, or ``'predict'``
        """
        if stage == "fit":
            dataloader_train = self._create_dataloader(
                self.dataset_train, is_training=True
            )
            self._dataloader_train = dataloader_train
            if self.dataset_val is not None:
                dataloader_val = self._create_dataloader(
                    self.dataset_val, is_training=False
                )
                self._dataloader_val = dataloader_val

        elif stage == "test":
            dataloader_test = self._create_dataloader(
                self.dataset_test, is_training=False
            )
            self._dataloader_test = dataloader_test

        elif stage == "predict":
            dataloader_predict = self._create_dataloader(
                self.dataset_predict, is_training=False
            )
            self._dataloader_predict = dataloader_predict

    def configure_label_encoder(self, params: dict):
        """Configure the label encoder with new parameters.

        :param dict params: parameters for label encoder configuration
        """
        # Add task_type if not provided (default to regression)
        if "encoder_type" not in params:
            params = params.copy()
            params["encoder_type"] = "regression"
        self._create_label_encoder(params)

    def configure_label_encoder_task(
        self,
        task_idx: int = 0,
        task_label: str = "output",
        class_thresholds: list[float] | None = None,
        class_labels: list[str] | None = None,
    ):
        """Configure a single task in the label encoder.

        :param int task_idx: index of the task to configure
        :param str task_label: human-readable label for the task
        :param list[float] | None class_thresholds: thresholds for binary classification
        :param list[str] | None class_labels: class label names
        """
        self._label_encoder._set_task_params(
            task_idx, task_label, class_thresholds, class_labels
        )

    def parse_output(
        self,
        output: np.ndarray,
        tag: str,
        convert_to_labels: bool = True,
    ):
        """Parse model output using the label encoder.

        :param np.ndarray output: raw model output array
        :param str tag: suffix for column names (e.g. ``'predictions'``)
        :param bool convert_to_labels: whether to convert to class labels
        :returns: labelled DataFrame with parsed outputs
        :rtype: pandas.DataFrame
        """
        return self._label_encoder.process(output, tag, convert_to_labels)

    def has_class_labels(self):
        """Check whether the label encoder has classification thresholds configured.

        :returns: True if class labels are configured and non-None
        :rtype: bool
        """
        if not hasattr(self._label_encoder.params, "class_thresholds"):
            return False
        if self._label_encoder.params.class_thresholds == {}:
            return False
        elif all(
            x is None for x in self._label_encoder.params.class_thresholds.values()
        ):
            return False
        else:
            return True

    def load_state_dict(self, state_dict):
        """Restore internal state from a serialized state dictionary.

        :param dict state_dict: dictionary containing y_scaler, label_encoder,
            and label_transform objects
        """
        if "datamodule_type" in state_dict:
            state_dict.pop("datamodule_type")

        self._y_scaler = state_dict["y_scaler"]
        self._label_encoder = state_dict["label_encoder"]
        self._label_transform = state_dict["label_transform"]
        # Get label encoder/transform params and create the appropriate encoder type
        # label_encoder_params = state_dict["label_encoder_params"]
        # label_transform_params = state_dict["label_transform_params"]

        # # Create encoder of the correct type and load its state
        # self._create_label_encoder(label_encoder_params)
        # self._create_label_transform(label_transform_params)


DataModuleRegistry = ClassRegistry[BaseDataModule]()
