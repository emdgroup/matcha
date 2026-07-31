"""Chemprop-compatible molecular featurization using message-passing neural networks."""

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.classic.tabular_datamodule import TabularDataModule
from matcha.utils.schemas.datamodules import ChempropDataModuleInputModel
from chemprop import data, featurizers
import numpy as np
from rdkit.Chem.rdchem import Mol
from matcha.utils import silence_nuisance_warnings
from torch import tensor, float32
from torch.utils.data import StackDataset


@DataModuleRegistry.register("chemprop")
class ChempropDataModule(TabularDataModule):
    """Chemprop message-passing featurization class.

    Extends :class:`TabularDataModule` to produce
    :class:`chemprop.data.MoleculeDataset` objects compatible with the Chemprop
    framework. Optionally includes tabular molecular features as extra
    descriptors (``x_d``).

    :param list[str] | None feature_list: list of tabular feature sets to
        compute as extra descriptors, or None to use only MPNN features
    :param dict | None engine_params: parameters for the RDKit engine
    """

    def __init__(
        self,
        feature_list: list[str] | None = None,
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
        # Handle the case where no features are used
        original_feature_list = feature_list  # Store the original value
        if feature_list is None:
            self._use_features = False
            feature_list = ["estate"]  # will be ignored, just for parent init
        else:
            self._use_features = True

        # Let parent class handle all initialization including params, input_dim, etc.
        super().__init__(
            feature_list,
            engine_params,
            is_classification,
            scaler_type,
            clip,
            label_encoder_params,
            label_transform_params,
            batch_size,
            num_workers,
            augment_resonance=augment_resonance,
        )

        # Override the feature_list in params to preserve the original None value
        self.params.feature_list = original_feature_list

        # ChemProp-specific initialization
        self._chemprop = featurizers.SimpleMoleculeMolGraphFeaturizer()
        params = self.params.model_dump()
        params["datamodule_type"] = "chemprop"
        params["label_encoder_params"] = self.params.label_encoder_params
        params["label_transform_params"] = self.params.label_transform_params
        self.params = ChempropDataModuleInputModel(**params)
        silence_nuisance_warnings()

    @property
    def use_features(self) -> bool:
        return self._use_features

    def generate_features(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generates unscaled features for ChemProp and creates a StackDataset.

        This method only handles feature generation without any X/Y scaling.
        Use featurize() for scaled features.

        :param mol_list: list of rdkit molecules to process
        :param y: array (N, X) of labels, or None
        :param bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None
        :param n_jobs: number of cores to use when featurizing the input,
            if None is passed a reasonable n_jobs value will be guessed from the
            amount of data
        :return: Unscaled StackDataset with all components for ChemProp
        """
        # validate inputs without scaling
        mol_list, y, bound_mask, n_jobs = self._validate_input(
            mol_list, y, bound_mask, n_jobs
        )

        # If using features, get tabular features from parent
        if self.use_features:
            tab_feats = super().generate_features(mol_list, y, bound_mask, n_jobs)
            # Extract the molecular features for x_d
            mol_features = tab_feats.datasets["mol_features"]
            y_values = tab_feats.datasets["y"]
        else:
            mol_features = None
            y_values = tensor(y, dtype=float32)

        # Create masks for ChemProp (will be None if no bound_mask provided)
        lt_masks = []
        gt_masks = []

        # Only process masks if bound_mask is provided
        if bound_mask is not None:
            if isinstance(bound_mask[0], str):
                for i, mask in enumerate(bound_mask):
                    if "<" in mask:
                        lt_masks.append(np.array([True]))
                        gt_masks.append(np.array([False]))
                    elif ">" in mask:
                        lt_masks.append(np.array([False]))
                        gt_masks.append(np.array([True]))
                    else:
                        lt_masks.append(np.array([False]))
                        gt_masks.append(np.array([False]))
            else:
                # Multi-task case: bound_mask is task-first, then sample
                # bound_mask[task_idx][sample_idx] gives mask for task_idx, sample_idx
                n_tasks = len(bound_mask)
                for i in range(len(mol_list)):
                    lt_task_masks = []
                    gt_task_masks = []
                    for task_idx in range(n_tasks):
                        mask = bound_mask[task_idx][i]
                        if "<" in mask:
                            lt_task_masks.append(True)
                            gt_task_masks.append(False)
                        elif ">" in mask:
                            lt_task_masks.append(False)
                            gt_task_masks.append(True)
                        else:
                            lt_task_masks.append(False)
                            gt_task_masks.append(False)
                    lt_masks.append(np.array(lt_task_masks))
                    gt_masks.append(np.array(gt_task_masks))
        else:
            # Create default masks (None for each molecule)
            lt_masks = [None] * len(mol_list)
            gt_masks = [None] * len(mol_list)

        # Create StackDataset with all components needed for ChemProp
        dataset_dict = {
            "mol": mol_list,
            "y": y_values,
            "lt_mask": lt_masks,
            "gt_mask": gt_masks,
        }

        if mol_features is not None:
            dataset_dict["mol_features"] = mol_features

        return StackDataset(**dataset_dict)

    def _create_dataloader(self, dataset, is_training):
        return data.build_dataloader(
            dataset,
            shuffle=is_training,
            batch_size=self.params.batch_size,
            num_workers=0,
        )

    def create_dataloader(self, dataset, is_training):
        return self._create_dataloader(dataset, is_training)

    def fit(self, dataset: StackDataset) -> None:
        """Fits scalers and other stateful transformations on the dataset.

        This method fits both X and Y scalers on the training data.

        :param StackDataset dataset: dataset to fit transformations on
        """
        # Fit X scaler if not frozen
        if not self._freeze_x_scaler and self.use_features:
            self._fit_x(dataset)

        # Fit Y scaler via parent class
        self._fit_y(dataset)

    def transform(self, dataset: StackDataset) -> StackDataset:
        """Applies fitted transformations to the dataset.

        This method applies previously fitted X and Y scalers to the dataset.

        :param StackDataset dataset: dataset to transform
        :return StackDataset: transformed dataset (modified in-place)
        """
        if self.use_features:
            self._transform_x(dataset)
        self._transform_y(dataset)

        return dataset

    def _stack_to_moleculedataset(self, dataset: StackDataset) -> data.MoleculeDataset:
        """Convert an internal StackDataset to a Chemprop MoleculeDataset.

        :param StackDataset dataset: dataset containing mol, y, masks, and
            optionally mol_features
        :returns: Chemprop-compatible dataset
        :rtype: chemprop.data.MoleculeDataset
        """
        # Extract components from StackDataset and create MoleculeDataset
        mol_list = dataset.datasets["mol"]
        y_values = dataset.datasets["y"].numpy()
        lt_masks = dataset.datasets["lt_mask"]
        gt_masks = dataset.datasets["gt_mask"]

        # Get scaled molecular features if they exist
        mol_features = (
            dataset.datasets["mol_features"]
            if "mol_features" in dataset.datasets
            else None
        )

        # Create datapoints with processed features and labels
        datapoints = []
        for i, (mol, y_val) in enumerate(zip(mol_list, y_values)):
            # Get features if needed
            x_d = (
                mol_features[i].detach().numpy()
                if self.use_features and mol_features is not None
                else None
            )

            # Create datapoint with proper mask handling
            datapoint = data.MoleculeDatapoint(
                mol=mol,
                y=y_val,
                x_d=x_d,
                lt_mask=lt_masks[i],
                gt_mask=gt_masks[i],
            )
            datapoints.append(datapoint)

        # Create and return processed MoleculeDataset
        return data.MoleculeDataset(datapoints)

    def featurize(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> data.MoleculeDataset:
        """Processes a list of N molecules and a numpy array (N,X) encoding the
        labels into a ChemProp MoleculeDataset, which can then be further processed
        for ChemProp neural network training.

        This method chains the unscaled feature generation with
        the necessary scaling operations.

        :param mol_list: list of N rdkit molecules to process
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
        :return: Processed MoleculeDataset ready for ChemProp neural network training
        """
        # Generate unscaled features in StackDataset format
        dataset = self.generate_features(mol_list, y, bound_mask, n_jobs)

        # Apply X processing (scaling) if needed
        if self.use_features:
            if is_training and not self._freeze_x_scaler:
                self._fit_x(dataset)
            self._transform_x(dataset)

        if is_training:
            self._fit_y(dataset)
        self._transform_y(dataset)

        # Apply Y processing for classification (censor mask already handled)
        if self.params.is_classification and self._label_encoder.is_set():
            stack = []
            y = dataset.datasets["y"].numpy()
            for i in range(y.shape[1]):
                ith = self._label_encoder._continuous_to_categorical(y[:, i], i)
                stack.append(ith)
            y = np.concatenate(stack, axis=1)
            # Update the dataset with the classification modifications
            dataset.datasets["y"] = tensor(y, dtype=float32)

        # Create and return processed MoleculeDataset
        return self._stack_to_moleculedataset(dataset)

    def state_dict(self):
        """Utility for MLFlow logging"""

        return {
            "ID": "Chemprop",
            "input_dim": self.params.input_dim,
            "params": self.params.model_dump(),
            "use_features": self.use_features,
            "y_scaler": self._y_scaler,
            "x_scaler": self._x_scaler,
            "label_encoder": self._label_encoder,
            "label_transform": self._label_transform,
        }

    def load_state_dict(self, state_dict):
        """Utility for MLFlow logging"""
        state_dict["params"]["datamodule_type"] = "tabular"
        if state_dict["params"]["feature_list"] is None:
            state_dict["params"]["feature_list"] = ["estate"]  # will be ignored
            super().load_state_dict(state_dict)
            self.params.feature_list = None
        else:
            super().load_state_dict(state_dict)

        params = self.params.model_dump()
        params["datamodule_type"] = "chemprop"
        self.params = ChempropDataModuleInputModel(**params)

        # Since parent handles params, we only need to restore chemprop-specific state
        self._use_features = state_dict["use_features"]

    @classmethod
    def dummy(cls):
        """Utility to make a dummy class with default params. Can be
        combined with load_state_dict to recreate a datamodule from
        a state dict
        """
        return cls(feature_list=["estate"])
