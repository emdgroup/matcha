"""On-the-fly DataModule that defers featurization to batch collation time."""

import inspect
import logging

import numpy as np
import scipy.sparse as sp
from matcha.datamodules.base_datamodule import BaseDataModule, DataModuleRegistry
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from torch.utils.data import StackDataset, DataLoader
from lightning import LightningDataModule

logger = logging.getLogger(__name__)


class OnTheFlyDataset:
    """Dataset that stores raw SMILES and labels for on-the-fly featurization.

    Used by :class:`OnTheFlyDataModule` to defer feature computation
    to collation time rather than computing all features upfront.

    :param smiles: list of SMILES strings
    :param y: label array, typically a sparse matrix of shape ``(N, T)``
    :param coords: optional list of N arrays, each ``(A_i, 3)`` of per-atom
        3D coordinates. When set, the wrapper forwards it to the base
        datamodule's ``generate_features(coords=...)`` at collation time
        (if the base accepts it) — otherwise it is silently dropped with a
        one-shot warning.
    """

    def __init__(
        self,
        smiles: list[str],
        y: np.ndarray,
        coords: list[np.ndarray] | None = None,
    ):
        self.smiles = smiles
        self.y = y
        self.coords = coords

    def __len__(self):
        """Return the number of samples."""
        return len(self.smiles)

    def __getitem__(self, idx):
        """Return a dict with ``smiles``, ``y`` (and ``coords`` when set) for the given index."""
        item = {"smiles": self.smiles[idx], "y": self.y[idx]}
        if self.coords is not None:
            item["coords"] = self.coords[idx]
        return item


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

        # Coords-capability probe state: filled on first collate that carries
        # a ``coords`` field. Cached because ``inspect.signature`` is not free
        # and we hit collate once per batch.
        self._base_accepts_coords: bool | None = None
        self._coords_ignore_warned = False

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
        train_coords: list[np.ndarray] | None = None,
        val_coords: list[np.ndarray] | None = None,
        test_coords: list[np.ndarray] | None = None,
        predict_coords: list[np.ndarray] | None = None,
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
        :param train_coords: optional training per-atom 3D coordinates
        :param val_coords: optional validation per-atom 3D coordinates
        :param test_coords: optional test per-atom 3D coordinates
        :param predict_coords: optional prediction per-atom 3D coordinates
        """
        if train_smiles is not None and train_y is not None:
            self._raw_train = OnTheFlyDataset(train_smiles, train_y, train_coords)
        if val_smiles is not None and val_y is not None:
            self._raw_val = OnTheFlyDataset(val_smiles, val_y, val_coords)
        if test_smiles is not None and test_y is not None:
            self._raw_test = OnTheFlyDataset(test_smiles, test_y, test_coords)
        if predict_smiles is not None:
            # For prediction, y can be None or dummy values
            if predict_y is None:
                predict_y = np.zeros(len(predict_smiles))
            self._raw_predict = OnTheFlyDataset(
                predict_smiles, predict_y, predict_coords
            )

    def _probe_base_accepts_coords(self) -> bool:
        """Return whether ``base.generate_features`` accepts a ``coords`` kwarg.

        Result is cached on ``self._base_accepts_coords`` — ``inspect.signature``
        is called at most once per instance.
        """
        if self._base_accepts_coords is None:
            try:
                sig = inspect.signature(self.base.generate_features)
                self._base_accepts_coords = "coords" in sig.parameters
            except (TypeError, ValueError):
                self._base_accepts_coords = False
        return self._base_accepts_coords

    def collate_fn(self, batch: list[dict]) -> dict:
        """Generate features on-the-fly from batch SMILES and collate.

        Converts SMILES to molecules, applies label encoding (sparse to dense,
        replacing 0 with NaN and -1 with 0), then delegates to the base
        datamodule's feature generation and collation. When batch items carry
        a ``coords`` field, the wrapper probes the base's ``generate_features``
        signature once (cached) and forwards coords via ``coords=`` if
        accepted. Otherwise coords are silently dropped and a single
        ``logging.warning`` is emitted so the pytest ``filterwarnings=error``
        gate is not tripped.

        :param batch: list of dicts with ``smiles``, ``y``, and optionally
            ``coords`` keys
        :return: collated batch dict from the base datamodule
        """
        has_coords = "coords" in batch[0]

        # Extract mols and labels from batch
        mols = [Chem.MolFromSmiles(item["smiles"]) for item in batch]
        y_batch = sp.vstack([item["y"] for item in batch]).toarray()

        y_batch[y_batch == 0] = np.nan
        y_batch[y_batch == -1] = 0

        # Optionally forward coords when the base datamodule accepts them
        forward_coords = None
        if has_coords:
            if self._probe_base_accepts_coords():
                forward_coords = [
                    np.asarray(item["coords"], dtype=np.float32) for item in batch
                ]
            elif not self._coords_ignore_warned:
                logger.warning(
                    "Coords were supplied to OnTheFlyDataModule but the base "
                    "datamodule (%s) does not accept a `coords` kwarg on "
                    "generate_features; coords will be ignored for the rest "
                    "of this run.",
                    type(self.base).__name__,
                )
                self._coords_ignore_warned = True

        # Generate features using the base datamodule
        if forward_coords is not None:
            features = self.base.generate_features(
                mols, y_batch, coords=forward_coords, n_jobs=1
            )
        else:
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
