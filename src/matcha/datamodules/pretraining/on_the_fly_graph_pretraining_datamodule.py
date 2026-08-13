"""On-the-fly DataModule variant for graph pretraining.

Delays graph featurization until batch collation time, saving memory when
working with large pretraining datasets. Follows the same wrapper pattern
as :class:`OnTheFlyMLMDataModule`.
"""

import numpy as np
from rdkit import Chem
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)


class OnTheFlyGraphPretrainingDataset:
    """Dataset that stores raw SMILES with molecule-level and atom-level labels.

    Used by :class:`OnTheFlyGraphPretrainingDataModule` to defer featurization
    to collation time.

    :param smiles: list of SMILES strings
    :param y_graph: array ``(N, G)`` of molecule-level targets
    :param y_node: list of N arrays, each ``(A_i, T)`` of atom-level targets
    :param coords: optional list of N arrays, each ``(A_i, 3)`` of per-atom
        3D coordinates in the input mol's atom order. When set, the wrapper
        forwards it to the base datamodule's ``generate_features(coords=...)``
        at batch collation time (e.g. for 3D pretraining).
    """

    def __init__(
        self,
        smiles: list[str],
        y_graph: np.ndarray,
        y_node: list[np.ndarray],
        coords: list[np.ndarray] | None = None,
    ):
        self.smiles = smiles
        self.y_graph = y_graph
        self.y_node = y_node
        self.coords = coords

    def __len__(self):
        """Return the number of samples."""
        return len(self.smiles)

    def __getitem__(self, idx):
        """Return a dict with ``smiles``, ``y_graph``, ``y_node`` (and ``coords`` when set) for the given index."""
        item = {
            "smiles": self.smiles[idx],
            "y_graph": self.y_graph[idx],
            "y_node": self.y_node[idx],
        }
        if self.coords is not None:
            item["coords"] = self.coords[idx]
        return item


@DataModuleRegistry.register("on_the_fly_graph_pretraining")
class OnTheFlyGraphPretrainingDataModule(LightningDataModule):
    """DataModule that delays graph pretraining feature generation until batch time.

    Wraps a :class:`GraphPretrainingDataModule` and stores raw SMILES,
    molecule-level labels, and atom-level labels.  Graph construction and
    featurization are performed inside the collate function when each batch
    is assembled, keeping memory usage low for large-scale pretraining.

    :param GraphPretrainingDataModule base: base datamodule used for
        featurization
    :param int num_workers: number of dataloader workers, defaults to 0
    """

    def __init__(
        self,
        base: GraphPretrainingDataModule,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.base = base
        self.params = base.params

        self._raw_train = None
        self._raw_val = None
        self._raw_test = None
        self._raw_predict = None

        self._dataloader_train = None
        self._dataloader_val = None
        self._dataloader_test = None
        self._dataloader_predict = None

        self.num_workers = num_workers

    # ------------------------------------------------------------------
    # Dataloader accessors
    # ------------------------------------------------------------------

    def train_dataloader(self):
        """Return the training dataloader."""
        return self._dataloader_train

    def val_dataloader(self):
        """Return the validation dataloader, or an empty one if not set."""
        if isinstance(self._dataloader_val, DataLoader):
            return self._dataloader_val
        return DataLoader({})

    def test_dataloader(self):
        """Return the test dataloader."""
        return self._dataloader_test

    def predict_dataloader(self):
        """Return the prediction dataloader."""
        return self._dataloader_predict

    # ------------------------------------------------------------------
    # Data setters
    # ------------------------------------------------------------------

    def set_data(
        self,
        train_smiles: list[str] | None = None,
        train_y_graph: np.ndarray | None = None,
        train_y_node: list[np.ndarray] | None = None,
        val_smiles: list[str] | None = None,
        val_y_graph: np.ndarray | None = None,
        val_y_node: list[np.ndarray] | None = None,
        test_smiles: list[str] | None = None,
        test_y_graph: np.ndarray | None = None,
        test_y_node: list[np.ndarray] | None = None,
        predict_smiles: list[str] | None = None,
        predict_y_graph: np.ndarray | None = None,
        predict_y_node: list[np.ndarray] | None = None,
        train_coords: list[np.ndarray] | None = None,
        val_coords: list[np.ndarray] | None = None,
        test_coords: list[np.ndarray] | None = None,
        predict_coords: list[np.ndarray] | None = None,
    ):
        """Set raw data for each split.

        :param train_smiles: training SMILES
        :param train_y_graph: training molecule-level labels ``(N, G)``
        :param train_y_node: training atom-level labels, list of ``(A_i, T)``
        :param val_smiles: validation SMILES
        :param val_y_graph: validation molecule-level labels
        :param val_y_node: validation atom-level labels
        :param test_smiles: test SMILES
        :param test_y_graph: test molecule-level labels
        :param test_y_node: test atom-level labels
        :param predict_smiles: prediction SMILES
        :param predict_y_graph: prediction molecule-level labels
        :param predict_y_node: prediction atom-level labels
        :param train_coords: optional training per-atom 3D coordinates
        :param val_coords: optional validation per-atom 3D coordinates
        :param test_coords: optional test per-atom 3D coordinates
        :param predict_coords: optional prediction per-atom 3D coordinates
        """
        if (
            train_smiles is not None
            and train_y_graph is not None
            and train_y_node is not None
        ):
            self._raw_train = OnTheFlyGraphPretrainingDataset(
                train_smiles, train_y_graph, train_y_node, train_coords
            )
        if (
            val_smiles is not None
            and val_y_graph is not None
            and val_y_node is not None
        ):
            self._raw_val = OnTheFlyGraphPretrainingDataset(
                val_smiles, val_y_graph, val_y_node, val_coords
            )
        if (
            test_smiles is not None
            and test_y_graph is not None
            and test_y_node is not None
        ):
            self._raw_test = OnTheFlyGraphPretrainingDataset(
                test_smiles, test_y_graph, test_y_node, test_coords
            )
        if (
            predict_smiles is not None
            and predict_y_graph is not None
            and predict_y_node is not None
        ):
            self._raw_predict = OnTheFlyGraphPretrainingDataset(
                predict_smiles, predict_y_graph, predict_y_node, predict_coords
            )

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def collate_fn(self, batch: list[dict]) -> dict:
        """Generate graph pretraining features on-the-fly from SMILES.

        Converts SMILES to molecules, delegates to the base datamodule's
        :meth:`generate_features`, then reshapes into the dict format
        expected by :class:`BaseGraphPretrainingModel`. When batch items
        carry per-atom ``coords`` (e.g. for 3D pretraining), they are
        forwarded to the base as ``generate_features(..., coords=...)``.

        :param batch: list of dicts with ``smiles``, ``y_graph``, ``y_node``,
            and optionally ``coords``
        :return: dict with ``graph``, ``y_node``, ``y_graph``
        """
        has_coords = "coords" in batch[0]

        mols = [Chem.MolFromSmiles(item["smiles"]) for item in batch]
        y_graph = np.array([item["y_graph"] for item in batch], dtype=np.float32)
        y_node = [np.asarray(item["y_node"], dtype=np.float32) for item in batch]
        coords = (
            [np.asarray(item["coords"], dtype=np.float32) for item in batch]
            if has_coords
            else None
        )

        # Filter out invalid molecules
        if coords is None:
            valid = [
                (m, yg, yn, None)
                for m, yg, yn in zip(mols, y_graph, y_node)
                if m is not None
            ]
        else:
            valid = [
                (m, yg, yn, ci)
                for m, yg, yn, ci in zip(mols, y_graph, y_node, coords)
                if m is not None
            ]
        if not valid:
            raise ValueError("No valid molecules in batch")

        mols_v, y_graph_v, y_node_v, coords_v = zip(*valid)
        mols = list(mols_v)
        y_graph_arr = np.array(y_graph_v)
        y_node_list = list(y_node_v)
        coords_list = list(coords_v) if has_coords else None

        # Generate features using the base datamodule then apply fitted scalers
        if coords_list is not None:
            features = self.base.generate_features(
                mols, y_graph_arr, y_node_list, coords=coords_list, n_jobs=1
            )
        else:
            features = self.base.generate_features(
                mols, y_graph_arr, y_node_list, n_jobs=1
            )
        self.base.transform(features)

        # Convert StackDataset to list-of-dicts and delegate to base collate
        batch_dicts = []
        for i in range(len(mols)):
            item = {}
            for key, dataset in features.datasets.items():
                item[key] = dataset[i]
            batch_dicts.append(item)

        return self.base.collate_fn(batch_dicts)

    # ------------------------------------------------------------------
    # Dataloader creation
    # ------------------------------------------------------------------

    def _create_dataloader(
        self,
        dataset: OnTheFlyGraphPretrainingDataset,
        is_training: bool,
    ) -> DataLoader:
        """Create a DataLoader with on-the-fly graph collation.

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

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def featurize(
        self, mol_list, y_graph=None, y_node=None, is_training=True, n_jobs=None
    ):
        """Delegate to base datamodule's featurize method.

        :param mol_list: list of RDKit molecules
        :param y_graph: molecule-level labels
        :param y_node: atom-level labels
        :param is_training: whether to fit scalers
        :param n_jobs: number of parallel workers
        :return: featurized StackDataset
        """
        return self.base.featurize(mol_list, y_graph, y_node, is_training, n_jobs)

    def generate_features(self, mol_list, y_graph=None, y_node=None, n_jobs=None):
        """Delegate to base datamodule's generate_features method.

        :param mol_list: list of RDKit molecules
        :param y_graph: molecule-level labels
        :param y_node: atom-level labels
        :param n_jobs: number of parallel workers
        :return: StackDataset of unscaled features
        """
        return self.base.generate_features(mol_list, y_graph, y_node, n_jobs)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str):
        """Setup dataloaders from raw data.

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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise state for MLFlow logging.

        :return: dict containing ID, base state, and num_workers
        """
        return {
            "ID": "on_the_fly_graph_pretraining",
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
