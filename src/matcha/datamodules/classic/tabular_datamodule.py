"""Tabular molecular featurization for MLP-based neural network training."""

from torch import tensor, float32
from torch.utils.data import StackDataset
from matcha.datamodules.classic.rdkit_engine import Engine as RDKitEngine
from matcha.datamodules.base_datamodule import BaseDataModule, DataModuleRegistry
from matcha.utils.schemas.datamodules import TabularDataModuleInputModel
from sklearn.preprocessing import StandardScaler
import numpy as np
from rdkit.Chem.rdchem import Mol
from matcha.utils.schemas import MolDataset

# Define which features belong to which engine
RDKIT_FEATURES = {
    "ecfp",
    "ecfp_count",
    "erg",
    "avalon",
    "estate",
    "rdkit_all_descriptors",
    "map4",
    "mhfp",
    "rdkit_fp",
    "pubchem_fp",
    "mordred",
}


@DataModuleRegistry.register("tabular")
class TabularDataModule(BaseDataModule):
    """Tabular molecular representation featurization class. Allows users to
    convert a list of rdkit molecules and labels into a TensorDataset ready
    to be used for MLP-like neural network training. It uses the
    :class:`mnet.datamodules.rdkit_engine.Engine` class for molecular fingerprints
    and descriptors. Common featurization logic is inherited from
    :class:`BaseDataModule`.

    The main purpose of the class is to enable the use of :method:`featurize`.
    Please check out :method:`featurize` for further information on the class' usage.

    :param list feature_list: list of strings defining which feature set
            to compute
    """

    def __init__(
        self,
        feature_list: list[str],
        engine_params: dict | None = None,
        is_classification: bool = False,
        scaler_type: str = "standard",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_params: dict = {},
        batch_size: int = 256,
        num_workers: int = 0,
        augment_resonance: bool = False,
    ):
        # Calculate input dimension
        input_dim = RDKitEngine().calculate_feature_dim(feature_list)

        self.params = TabularDataModuleInputModel(
            input_dim=input_dim,
            feature_list=feature_list,
            engine_params=engine_params,
            is_classification=is_classification,
            scaler_type=scaler_type,
            clip=clip,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )

        super(TabularDataModule, self).__init__(
            scaler_type=scaler_type,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            augment_resonance=augment_resonance,
        )

        self._x_scaler = StandardScaler()
        self._freeze_x_scaler = False

    def _validate_datamodule_input(self, x, y, bound_mask):
        """Simple validation function to check that the input dataset
        has adequate types. In case it doesn't, either throw an error or fix it."""
        valid = MolDataset(mols=x, y=y, bound_mask=bound_mask)
        return valid.mols, valid.y, valid.bound_mask

    @property
    def x_scaler(self) -> StandardScaler:
        """Property to access the feature scaler instance."""
        return self._x_scaler

    def generate_features(
        self,
        input_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generates unscaled features for molecules and creates a StackDataset.

        This method only handles feature generation without any scaling.
        Use featurize() for scaled features.

        :param input_list: list of rdkit molecules to process
        :param y: array (N, X) of labels, or None
        :param bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None
        :param n_jobs: number of cores to use when featurizing the input,
            if None is passed a reasonable n_jobs value will be guessed from the
            amount of data
        :return: Unscaled StackDataset with keys `mol_features` and `y`
        """
        # validate inputs without scaling
        input_list, y, bound_mask, n_jobs = self._validate_input(
            input_list, y, bound_mask, n_jobs
        )

        # compute features using RDKit engine
        rdkit_engine = RDKitEngine(n_jobs)
        if self.params.engine_params is not None:
            for key in self.params.engine_params:
                if key.lower() in RDKIT_FEATURES:
                    rdkit_engine.set_defaults(key, self.params.engine_params[key])
        x = rdkit_engine.get_features(input_list, self.params.feature_list)

        x_tensor = tensor(x, dtype=float32)
        y_tensor = tensor(y, dtype=float32)

        return StackDataset(mol_features=x_tensor, y=y_tensor)

    def _invert_x(self, dataset: StackDataset) -> None:
        """Utility to invert X scaling, modifying dataset in-place.

        :param dataset: StackDataset to invert
        """
        features = dataset.datasets["mol_features"].numpy()
        inverted_features = self._x_scaler.inverse_transform(features)
        dataset.datasets["mol_features"] = tensor(inverted_features, dtype=float32)

    def _fit_x(self, dataset: StackDataset) -> None:
        """Utility to fit and transform X, modifying dataset in-place.

        :param dataset: StackDataset to fit and transform
        """
        features = dataset.datasets["mol_features"].numpy()
        self._x_scaler.fit(features)

    def _transform_x(self, dataset: StackDataset) -> None:
        """Utility to transform X, modifying dataset in-place.

        :param dataset: StackDataset to transform
        """
        features = dataset.datasets["mol_features"].numpy()
        scaled_features = self._x_scaler.transform(features)
        dataset.datasets["mol_features"] = tensor(scaled_features, dtype=float32)

    def fit(self, dataset: StackDataset) -> None:
        """Fits scalers and other stateful transformations on the dataset.

        This method fits both X and Y scalers on the training data.

        :param StackDataset dataset: dataset to fit transformations on
        """
        # Fit X scaler if not frozen
        if not self._freeze_x_scaler:
            self._fit_x(dataset)

        # Fit Y scaler via parent class
        self._fit_y(dataset)

    def transform(self, dataset: StackDataset) -> StackDataset:
        """Applies fitted transformations to the dataset.

        This method applies previously fitted X and Y scalers to the dataset.

        :param StackDataset dataset: dataset to transform
        :return StackDataset: transformed dataset (modified in-place)
        """
        self._transform_x(dataset)
        self._transform_y(dataset)

        return dataset

    def featurize(
        self,
        input_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Processes a list of N molecules and a numpy array (N,X) encoding the
        labels into a TensorDataset, which can then be further processed for
        MLP-like neural network training.

        This method chains the unscaled feature generation with
        the necessary scaling operations.

        Example usage:

        .. code-block:: python
            TF = TabularDataModule(['ecfp', 'erg'])
            train_dataset = TF.featurize(train_mols, train_y, is_training=True)
            test_dataset = TF.featurize(test_mols, test_y, is_training=False)

        :param input_list: list of N rdkit molecules to process

        :param y: array (N, X), where X is the number of classes or
            endpoints, or None

        :param bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None

        :param is_training: Whether to fit the X and Y scalers on the input,
            or leverage pre-existing ones to only normalize

        :param n_jobs: number of cores to use when featurizing the input,
            if None is passed a reasonable n_jobs value will be guessed from the
            amount of data, defaults to None

        :return: Processed dataset providing keys `mol_features` and `y` ready for MLP-like neural network training
        """
        # Apply augmentation if enabled
        if is_training and self._augment_resonance:
            input_list, y, bound_mask = self.augment(
                input_list,
                y,
                bound_mask=bound_mask,
                use_resonance=self._augment_resonance,
                n_jobs=n_jobs,
            )

        # Generate unscaled features
        dataset = self.generate_features(input_list, y, bound_mask, n_jobs)

        # Apply scaling based on is_training flag
        if is_training:
            self.fit(dataset)

        self.transform(dataset)

        # Handle bound mask and classification transformations (only if not regression)
        self._process_y(dataset, bound_mask)

        return dataset

    def state_dict(self):
        """Utility for MLFlow logging"""

        return {
            "ID": "tabular",
            "input_dim": self.params.input_dim,
            "params": self.params.model_dump(),
            "y_scaler": self._y_scaler,
            "x_scaler": self._x_scaler,
            "label_encoder": self._label_encoder,
            "label_transform": self._label_transform,
        }

    def load_state_dict(self, state_dict):
        """Utility for MLFlow logging"""

        super().load_state_dict(state_dict)

        # Create InputModel with the remaining parameters
        self.params = TabularDataModuleInputModel(**state_dict["params"])

        self._x_scaler = state_dict["x_scaler"]

    @classmethod
    def dummy(cls):
        """Utility to make a dummy class with default params. Can be
        combined with load_state_dict to recreate a datamodule from
        a state dict
        """
        return cls(feature_list=["ECFP"])
