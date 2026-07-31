"""On-the-fly DataModule variant for MLM pretraining."""

from rdkit import Chem
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from matcha.datamodules.pretraining.clm_mlm_datamodule import CLMMLMDataModule
from matcha.datamodules.base_datamodule import DataModuleRegistry


class OnTheFlyMLMDataset:
    """Dataset that stores raw SMILES for on-the-fly MLM featurization.

    Unlike OnTheFlyDataset, this does not require labels or descriptors
    since MLM is self-supervised.
    """

    def __init__(self, smiles: list[str]):
        """Initialise with a list of SMILES strings.

        :param smiles: list of SMILES strings
        """
        self.smiles = smiles

    def __len__(self):
        """Return the number of samples."""
        return len(self.smiles)

    def __getitem__(self, idx):
        """Return a dict with ``smiles`` for the given index."""
        return {"smiles": self.smiles[idx]}


@DataModuleRegistry.register("on_the_fly_mlm")
class OnTheFlyMLMDataModule(LightningDataModule):
    """DataModule that delays MLM feature generation until batch iteration time.

    This wrapper takes a CLMMLMDataModule and modifies its behavior to generate
    features on-the-fly during batch collation instead of upfront during setup.
    This is useful for large-scale pretraining where memory is constrained.

    The collate function converts SMILES to molecules, generates tokenized
    features, and applies masking on-the-fly.

    :param CLMMLMDataModule base: The base MLM datamodule to use for featurization
    :param int num_workers: Number of dataloader workers, defaults to 0
    """

    def __init__(self, base: CLMMLMDataModule, num_workers: int = 0, **kwargs):
        """Initialise the on-the-fly MLM wrapper.

        :param base: base MLM datamodule used for tokenization and masking
        :param num_workers: number of dataloader workers, defaults to 0
        """
        super().__init__()

        self.base = base
        self.params = base.params

        # Copy collate function mappings from base if available
        self.collate_fn_map = getattr(base, "collate_fn_map", {}).copy()

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
        val_smiles: list[str] = None,
        test_smiles: list[str] = None,
        predict_smiles: list[str] = None,
    ):
        """Set the raw SMILES data for each split.

        For MLM, we only need SMILES - no labels required.

        :param train_smiles: Training SMILES
        :param val_smiles: Validation SMILES
        :param test_smiles: Test SMILES
        :param predict_smiles: Prediction SMILES
        """
        if train_smiles is not None:
            self._raw_train = OnTheFlyMLMDataset(train_smiles)
        if val_smiles is not None:
            self._raw_val = OnTheFlyMLMDataset(val_smiles)
        if test_smiles is not None:
            self._raw_test = OnTheFlyMLMDataset(test_smiles)
        if predict_smiles is not None:
            self._raw_predict = OnTheFlyMLMDataset(predict_smiles)

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate function that generates MLM features on-the-fly from batch SMILES.

        Converts SMILES to molecules, generates tokenized features with masking,
        and returns a batch ready for the MLM model.

        :param batch: List of dicts with 'smiles' key
        :return: Dict with 'token_ids', 'y', and 'mask' tensors
        """
        # Extract molecules from batch SMILES
        mols = [Chem.MolFromSmiles(item["smiles"]) for item in batch]

        # Filter out invalid molecules
        valid_mols = [mol for mol in mols if mol is not None]

        if len(valid_mols) == 0:
            raise ValueError("No valid molecules in batch")

        # Generate MLM features using the base datamodule
        # Note: is_training=False to avoid refitting dictionary
        # augment=True to still apply SMILES augmentation
        features = self.base.generate_features(
            valid_mols,
            y=None,
            bound_mask=None,
            augment=True,
            is_training=False,
            n_jobs=1,
        )

        # Convert StackDataset to dict for batching
        return {
            "token_ids": features.datasets["token_ids"],
            "y": features.datasets["y"],
            "mask": features.datasets["mask"],
        }

    def _create_dataloader(
        self, dataset: OnTheFlyMLMDataset, is_training: bool
    ) -> DataLoader:
        """Create a DataLoader with on-the-fly MLM collation.

        :param dataset: raw dataset of SMILES
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

    def featurize(
        self,
        mol_list,
        y=None,
        bound_mask=None,
        is_training: bool = True,
        n_jobs: int | None = None,
    ):
        """Delegate to base datamodule's featurize method.

        :param mol_list: list of RDKit molecules
        :param y: ignored (MLM is self-supervised)
        :param bound_mask: ignored for MLM
        :param is_training: whether to commit the dictionary
        :param n_jobs: number of parallel workers
        :return: StackDataset with ``token_ids``, ``y``, and ``mask``
        """
        return self.base.featurize(mol_list, y, bound_mask, is_training, n_jobs=n_jobs)

    def generate_features(
        self,
        mol_list,
        y=None,
        bound_mask=None,
        augment: bool = True,
        is_training: bool = True,
        n_jobs: int | None = None,
    ):
        """Delegate to base datamodule's generate_features method.

        :param mol_list: list of RDKit molecules
        :param y: ignored (MLM is self-supervised)
        :param bound_mask: ignored for MLM
        :param augment: whether to apply SMILES augmentation
        :param is_training: whether to update the dictionary
        :param n_jobs: number of parallel workers
        :return: StackDataset with ``token_ids``, ``y``, and ``mask``
        """
        return self.base.generate_features(
            mol_list, y, bound_mask, augment, is_training, n_jobs
        )

    def fit_dictionary(self, smiles_list: list[str], n_jobs: int = 4):
        """Pre-fit the dictionary on a set of SMILES before using on-the-fly loading.

        This is important for on-the-fly usage where we need a consistent
        dictionary across all batches.

        :param smiles_list: List of SMILES to fit dictionary on
        :param n_jobs: Number of parallel jobs
        """
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        valid_mols = [mol for mol in mols if mol is not None]

        # Fit dictionary using the base datamodule
        self.base.featurize(
            valid_mols,
            y=None,
            is_training=True,
            augment=False,
            n_jobs=n_jobs,
        )

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

    def state_dict(self) -> dict:
        """Serialise state for MLFlow logging.

        :return: dict containing ID, base state, and num_workers
        """
        return {
            "ID": "on_the_fly_mlm",
            "base_state_dict": self.base.state_dict(),
            "num_workers": self.num_workers,
        }

    def load_state_dict(self, state_dict: dict):
        """Restore state from a previously serialised dict.

        :param state_dict: dict produced by :meth:`state_dict`
        """
        self.base.load_state_dict(state_dict["base_state_dict"])
        self.num_workers = state_dict.get("num_workers", 0)
        self.params = self.base.params
