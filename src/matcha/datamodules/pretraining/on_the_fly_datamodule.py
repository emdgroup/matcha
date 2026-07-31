"""On-the-fly DataModule that defers featurization to batch collation time."""

import numpy as np
import scipy.sparse as sp
from matcha.datamodules.base_datamodule import BaseDataModule, DataModuleRegistry
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from torch.utils.data import StackDataset, DataLoader
from lightning import LightningDataModule


class OnTheFlyDataset:
    """Dataset that stores raw SMILES and labels for on-the-fly featurization.

    Used by :class:`OnTheFlyDataModule` to defer feature computation
    to collation time rather than computing all features upfront.

    :param smiles: list of SMILES strings
    :param y: label array, typically a sparse matrix of shape ``(N, T)``
    """

    def __init__(self, smiles: list[str], y: np.ndarray):
        self.smiles = smiles
        self.y = y

    def __len__(self):
        """Return the number of samples."""
        return len(self.smiles)

    def __getitem__(self, idx):
        """Return a dict with ``smiles`` and ``y`` for the given index."""
        return {"smiles": self.smiles[idx], "y": self.y[idx]}


@DataModuleRegistry.register("on_the_fly")
class OnTheFlyDataModule(LightningDataModule):
    """DataModule that delays feature generation until batch iteration time.

    This wrapper takes any base datamodule and modifies its behavior to generate
    features on-the-fly during batch collation instead of upfront during setup.
    This can be useful for saving memory or when features need to be computed
    dynamically.
    """

    def __init__(self, base: BaseDataModule, num_workers: int = 0, **kwargs):
        """Initialise the on-the-fly wrapper around a base datamodule.

        :param base: base datamodule used for feature generation and collation
        :param num_workers: number of dataloader workers, defaults to 0
        """
        super().__init__()

        self.base = base
        # Copy relevant parameters from base
        self.params = base.params

        # Copy collate function mappings
        self.collate_fn_map = base.collate_fn_map.copy()

        # Store raw data for on-the-fly processing
        self._raw_train = None
        self._raw_val = None
        self._raw_test = None
        self._raw_predict = None

        self._dataloader_train = None
        self._dataloader_val = None
        self._dataloader_test = None
        self._dataloader_predict = None

        self.num_workers = num_workers

    def train_dataloader(self):
        """Return the training dataloader."""
        return self._dataloader_train

    def val_dataloader(self):
        """Return the validation dataloader, or an empty one if not set."""
        if isinstance(self._dataloader_val, DataLoader):
            return self._dataloader_val
        else:
            return DataLoader({})

    def test_dataloader(self):
        """Return the test dataloader."""
        return self._dataloader_test

    def predict_dataloader(self):
        """Return the prediction dataloader."""
        return self._dataloader_predict

    def set_data(
        self,
        train_smiles: list[str] = None,
        train_y: np.ndarray = None,
        val_smiles: list[str] = None,
        val_y: np.ndarray = None,
        test_smiles: list[str] = None,
        test_y: np.ndarray = None,
        predict_smiles: list[str] = None,
        predict_y: np.ndarray = None,
    ):
        """Set the raw data for each split.

        :param train_smiles: training SMILES
        :param train_y: training labels (sparse matrix)
        :param val_smiles: validation SMILES
        :param val_y: validation labels (sparse matrix)
        :param test_smiles: test SMILES
        :param test_y: test labels (sparse matrix)
        :param predict_smiles: prediction SMILES
        :param predict_y: prediction labels (optional, defaults to zeros)
        """
        if train_smiles is not None and train_y is not None:
            self._raw_train = OnTheFlyDataset(train_smiles, train_y)
        if val_smiles is not None and val_y is not None:
            self._raw_val = OnTheFlyDataset(val_smiles, val_y)
        if test_smiles is not None and test_y is not None:
            self._raw_test = OnTheFlyDataset(test_smiles, test_y)
        if predict_smiles is not None:
            # For prediction, y can be None or dummy values
            if predict_y is None:
                predict_y = np.zeros(len(predict_smiles))
            self._raw_predict = OnTheFlyDataset(predict_smiles, predict_y)

    def collate_fn(self, batch: list[dict]) -> dict:
        """Generate features on-the-fly from batch SMILES and collate.

        Converts SMILES to molecules, applies label encoding (sparse to dense,
        replacing 0 with NaN and -1 with 0), then delegates to the base
        datamodule's feature generation and collation.

        :param batch: list of dicts with ``smiles`` and ``y`` keys
        :return: collated batch dict from the base datamodule
        """
        # Extract mols and labels from batch
        mols = [Chem.MolFromSmiles(item["smiles"]) for item in batch]
        y_batch = sp.vstack([item["y"] for item in batch]).toarray()

        y_batch[y_batch == 0] = np.nan
        y_batch[y_batch == -1] = 0

        # Generate features using the base datamodule
        features = self.base.generate_features(mols, y_batch, n_jobs=1)

        # Convert to list of dicts for collation
        batch_dicts = []
        for i in range(len(mols)):
            item = {}
            for key, dataset in features.datasets.items():
                if key == "y":
                    item[key] = dataset[i]
                else:
                    item[key] = dataset[i]
            batch_dicts.append(item)

        # Use base datamodule's collate function
        return self.base.collate_fn(batch_dicts)

    def _create_dataloader(
        self, dataset: OnTheFlyDataset, is_training: bool
    ) -> DataLoader:
        """Create a DataLoader with on-the-fly collation.

        :param dataset: raw dataset of SMILES and labels
        :param is_training: whether to shuffle the data
        :return: configured DataLoader
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.params.batch_size,
            shuffle=is_training,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    # Implement abstract methods from BaseDataModule
    def featurize(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Delegate to base datamodule's featurize method.

        :param mol_list: list of RDKit molecules
        :param y: label array
        :param bound_mask: optional bound mask for censored labels
        :param is_training: whether to fit scalers
        :param n_jobs: number of parallel workers
        :return: featurized StackDataset
        """
        return self.base.featurize(mol_list, y, bound_mask, is_training, n_jobs)

    def generate_features(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Delegate to base datamodule's generate_features method.

        :param mol_list: list of RDKit molecules
        :param y: label array
        :param bound_mask: optional bound mask for censored labels
        :param n_jobs: number of parallel workers
        :return: StackDataset of unscaled features
        """
        return self.base.generate_features(mol_list, y, bound_mask, n_jobs)

    def setup(self, stage: str):
        """Setup dataloaders using raw data instead of pre-computed features.

        :param stage: one of ``"fit"``, ``"test"``, or ``"predict"``
        """
        if stage == "fit":
            if self._raw_train is not None:
                self._dataloader_train = self._create_dataloader(
                    self._raw_train, is_training=True
                )
            if self._raw_val is not None:
                self._dataloader_val = self._create_dataloader(
                    self._raw_val, is_training=False
                )

        elif stage == "test":
            if self._raw_test is not None:
                self._dataloader_test = self._create_dataloader(
                    self._raw_test, is_training=False
                )

        elif stage == "predict":
            if self._raw_predict is not None:
                self._dataloader_predict = self._create_dataloader(
                    self._raw_predict, is_training=False
                )
