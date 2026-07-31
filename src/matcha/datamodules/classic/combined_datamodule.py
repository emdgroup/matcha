"""Combined data module for multi-modal molecular representation learning."""

from collections.abc import Callable
from typing import Any

from matcha.datamodules.base_datamodule import BaseDataModule, DataModuleRegistry
from matcha.datamodules.classic.tabular_datamodule import TabularDataModule
from matcha.datamodules.classic.graph_datamodule import GraphDataModule
from matcha.datamodules.utils import CombinedStackDataset, collate_fns
import inspect
import numpy as np
from rdkit.Chem.rdchem import Mol
from matcha.utils.schemas.datamodules import CombinedDataModuleInputModel


def default_merge(args):
    """Default merge function that returns the first value when a key appears in multiple datasets.

    :param list args: list of values from each dataset for the same key
    :returns: the first value
    """
    return args[0]


@DataModuleRegistry.register("combined")
class CombinedDataModule(BaseDataModule):
    """A DataModule that can combine multiple datamodules.

    This class is useful when you have multiple datamodules that you want to combine into a single
    datamodules.

    If a key appears in multiple datamodules, the values get merged according to merge_fn.

    :param datamodules: a list of datamodules to combine
    :param merge_fn: a dictionary mapping keys to merge functions
    """

    def __init__(
        self,
        datamodules: list[BaseDataModule],
        merge_fn: dict[str, Callable] | None = {"y": default_merge},
        is_classification: bool = False,
        scaler_type: str = "standard",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_params: dict = {},
        batch_size: int = 256,
        num_workers: int = 0,
        augment_resonance: bool = False,
    ):
        self.params = CombinedDataModuleInputModel(
            datamodules=[x.params.datamodule_type for x in datamodules],
            merge_fn=merge_fn,
            is_classification=is_classification,
            scaler_type=scaler_type,
            clip=clip,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        super().__init__(
            scaler_type=scaler_type,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            augment_resonance=augment_resonance,
        )

        self.datamodules = datamodules

        for i, feat in enumerate(self.datamodules):
            if isinstance(feat, TabularDataModule):
                self.tabular_datamodule_idx = i

    def generate_features(
        self,
        x: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> CombinedStackDataset:
        """Generate unscaled features for all sub-datamodules and combine them.
        Calls parent generate_features for each datamodule.

        Uses keyword arguments so that sub-datamodules with additional parameters
        (e.g. CLMDataModule's ``augment`` / ``is_training``) receive the correct
        values instead of swallowing ``n_jobs`` into a wrong positional slot.

        ``is_training`` is forwarded to sub-datamodules whose ``generate_features``
        accepts it (currently only :class:`CLMDataModule`), enabling the correct
        augmentation count (train vs. test) inside a :class:`CombinedDataModule`.
        """
        datasets = []
        for datamodule in self.datamodules:
            sig = inspect.signature(datamodule.generate_features)
            kwargs = dict(y=y, bound_mask=bound_mask, n_jobs=n_jobs)
            if "is_training" in sig.parameters:
                kwargs["is_training"] = is_training
            datasets.append(datamodule.generate_features(x, **kwargs))
        combined_dataset = CombinedStackDataset(datasets, self.params.merge_fn)
        self._keys_to_dataset_idx = combined_dataset._keys_to_dataset_idx
        return combined_dataset

    def featurize(
        self,
        x: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> CombinedStackDataset:
        """Featurize molecules using all sub-datamodules and apply scaling.

        :param list[Mol] x: list of rdkit molecules to process
        :param np.ndarray | None y: labels array or None
        :param list[str] | None bound_mask: bound mask or None
        :param bool is_training: whether to fit scalers on the input
        :param int | None n_jobs: number of parallel jobs
        :returns: combined dataset with features from all sub-datamodules
        :rtype: CombinedStackDataset
        """

        if is_training and self._augment_resonance:
            x, y, bound_mask = self.augment(
                x,
                y,
                bound_mask=bound_mask,
                use_resonance=self._augment_resonance,
                n_jobs=n_jobs,
            )

        dataset = self.generate_features(
            x, y, bound_mask, is_training=is_training, n_jobs=n_jobs
        )

        # Apply scaling based on is_training flag
        if is_training:
            self.fit(dataset)

        self.transform(dataset)

        # Handle bound mask and classification transformations (only if not regression)
        if bound_mask is not None or self.params.is_classification:
            self._process_y(dataset, bound_mask)

        return dataset

    def fit(self, dataset: CombinedStackDataset) -> None:
        """Fits scalers and other stateful transformations on the dataset.

        For CombinedDataModule, this calls fit on all sub-datamodules with their
        respective individual datasets.

        :param CombinedStackDataset dataset: dataset to fit transformations on
        """
        # For each sub-datamodule, call its fit method on its own dataset
        for i, datamodule in enumerate(self.datamodules):
            individual_dataset = dataset.datasets[i]
            datamodule.fit(individual_dataset)

    def transform(self, dataset: CombinedStackDataset) -> CombinedStackDataset:
        """Applies fitted transformations to the dataset.

        For CombinedDataModule, this calls transform on all sub-datamodules with
        their respective individual datasets.

        :param CombinedStackDataset dataset: dataset to transform
        :return CombinedStackDataset: transformed dataset (modified in-place)
        """
        # For each sub-datamodule, call its transform method on its own dataset
        for i, datamodule in enumerate(self.datamodules):
            individual_dataset = dataset.datasets[i]
            datamodule.transform(individual_dataset)

        return dataset

    def _process_y(self, dataset, bound_mask=None):
        """Process Y for all sub-datamodules in their individual datasets."""
        # Pass the first datamodule's individual dataset to its _process_y method
        first_datamodule_dataset = dataset.datasets[0]
        self.datamodules[0]._process_y(first_datamodule_dataset, bound_mask)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of items.

        :param batch: a list of items to collate

        :return: the collated batch
        """
        collated_batch = {}
        for key in self._keys_to_dataset_idx.keys():
            collate_fn = collate_fns[key]
            collated_batch[key] = collate_fn([item[key] for item in batch])
        return collated_batch

    def invert_y(self, y: np.ndarray) -> np.ndarray:
        # TODO (TP): For now assume that all datamodules have the same invert_y function, so just pick the first one.
        # For the future think about separating out the label transforms in a separate datamodule that can be combined with molecular datamodules.
        return self.datamodules[0].invert_y(y)

    def _transform_y(self, dataset: CombinedStackDataset) -> None:
        # See limitations above - delegate to first datamodule using its individual dataset
        first_datamodule_dataset = dataset.datasets[0]
        self.datamodules[0]._transform_y(first_datamodule_dataset)

    def parse_output(
        self, output: np.ndarray, tag: str, convert_to_labels: bool = True
    ):
        # See limitations above
        return self.datamodules[0]._label_encoder.process(
            output, tag, convert_to_labels
        )

    def configure_label_encoder(self, params):
        # See limitations above
        self.datamodules[0].configure_label_encoder(params)

    def has_class_labels(self):
        return self.datamodules[0].has_class_labels()

    def _invert_x(self, dataset: CombinedStackDataset) -> None:
        """Invert X scaling for the tabular datamodule component."""
        if hasattr(self, "tabular_datamodule_idx"):
            tabular_dataset = dataset.datasets[self.tabular_datamodule_idx]
            self.datamodules[self.tabular_datamodule_idx]._invert_x(tabular_dataset)

    def _fit_x(self, dataset: CombinedStackDataset) -> None:
        """Fit X for the tabular datamodule component."""
        if hasattr(self, "tabular_datamodule_idx"):
            tabular_dataset = dataset.datasets[self.tabular_datamodule_idx]
            self.datamodules[self.tabular_datamodule_idx]._fit_x(tabular_dataset)

    def _transform_x(self, dataset: CombinedStackDataset) -> None:
        """Transform X for the tabular datamodule component."""
        if hasattr(self, "tabular_datamodule_idx"):
            tabular_dataset = dataset.datasets[self.tabular_datamodule_idx]
            self.datamodules[self.tabular_datamodule_idx]._transform_x(tabular_dataset)

    def state_dict(self) -> dict:
        """Utility for MLFlow logging"""

        key_list = [x.__class__.__name__ for x in self.datamodules]
        dict_list = [x.state_dict() for x in self.datamodules]
        out_dict = dict(zip(key_list, dict_list))
        out_dict["ID"] = "combined"
        out_dict["params"] = self.params.model_dump()
        out_dict["keys_collate"] = self._keys_to_dataset_idx
        return out_dict

    def load_state_dict(self, state_dict: dict):
        """Utility for MLFlow logging"""

        datamodules = []
        for i, key in enumerate(state_dict.keys()):
            if key != "ID" and key != "keys_collate" and key != "params":
                dm = DataModuleRegistry[state_dict[key]["ID"]].dummy()
                dm.load_state_dict(state_dict[key])
                datamodules.append(dm)

        # Extract parameters and include the reconstructed datamodules
        params_dict = state_dict["params"].copy()
        params_dict["datamodules"] = [x.params.datamodule_type for x in datamodules]

        # Create InputModel with all parameters including datamodules
        self.params = CombinedDataModuleInputModel(**params_dict)
        self.datamodules = datamodules
        # Set the keys_to_dataset_idx mapping
        self._keys_to_dataset_idx = state_dict["keys_collate"]

    @classmethod
    def dummy(cls):
        """Utility to make a dummy class with default params. Can be
        combined with load_state_dict to recreate a datamodule from
        a state dict
        """
        tab = TabularDataModule(["ECFP"])
        graph = GraphDataModule()
        return cls([tab, graph])
