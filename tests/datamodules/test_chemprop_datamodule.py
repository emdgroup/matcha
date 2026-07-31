"""Tests for ChempropDataModule."""

from chemprop import data as chemprop_data
from torch.utils.data import StackDataset

from matcha.datamodules.classic.chemprop_datamodule import ChempropDataModule
from matcha.datamodules.base_datamodule import DataModuleRegistry


# ===================================================================
# Construction
# ===================================================================


class TestChempropDataModuleInit:
    def test_default_no_features(self):
        dm = ChempropDataModule()
        assert dm.params.datamodule_type == "chemprop"
        assert dm.use_features is False
        assert dm.params.feature_list is None

    def test_with_features(self):
        dm = ChempropDataModule(feature_list=["estate"])
        assert dm.use_features is True
        assert dm.params.feature_list == ["estate"]

    def test_registry_has_chemprop(self):
        assert "chemprop" in DataModuleRegistry


# ===================================================================
# generate_features
# ===================================================================


class TestChempropGenerateFeatures:
    def test_generate_features_no_features(self, small_mol_list, small_regression_y):
        dm = ChempropDataModule()
        ds = dm.generate_features(small_mol_list, small_regression_y, n_jobs=1)
        assert isinstance(ds, StackDataset)
        item = ds[0]
        assert "mol" in item
        assert "y" in item
        assert "lt_mask" in item
        assert "gt_mask" in item

    def test_generate_features_with_features(self, small_mol_list, small_regression_y):
        dm = ChempropDataModule(feature_list=["estate"])
        ds = dm.generate_features(small_mol_list, small_regression_y, n_jobs=1)
        assert "mol_features" in ds.datasets

    def test_generate_features_without_y(self, small_mol_list):
        dm = ChempropDataModule()
        ds = dm.generate_features(small_mol_list, None, n_jobs=1)
        assert ds.datasets["y"].shape[0] == len(small_mol_list)


class TestChempropGenerateFeaturesWithBoundMask:
    def test_bound_mask_less_than(self, small_mol_list, small_regression_y):
        masks = ["<"] * len(small_mol_list)
        dm = ChempropDataModule()
        ds = dm.generate_features(
            small_mol_list, small_regression_y, bound_mask=masks, n_jobs=1
        )
        lt_mask = ds.datasets["lt_mask"][0]
        assert lt_mask is not None
        assert lt_mask[0]

    def test_bound_mask_greater_than(self, small_mol_list, small_regression_y):
        masks = [">"] * len(small_mol_list)
        dm = ChempropDataModule()
        ds = dm.generate_features(
            small_mol_list, small_regression_y, bound_mask=masks, n_jobs=1
        )
        gt_mask = ds.datasets["gt_mask"][0]
        assert gt_mask is not None
        assert gt_mask[0]

    def test_bound_mask_exact(self, small_mol_list, small_regression_y):
        masks = ["="] * len(small_mol_list)
        dm = ChempropDataModule()
        ds = dm.generate_features(
            small_mol_list, small_regression_y, bound_mask=masks, n_jobs=1
        )
        lt_mask = ds.datasets["lt_mask"][0]
        gt_mask = ds.datasets["gt_mask"][0]
        assert not lt_mask[0]
        assert not gt_mask[0]


# ===================================================================
# featurize
# ===================================================================


class TestChempropFeaturize:
    def test_featurize_returns_molecule_dataset(
        self, small_mol_list, small_regression_y
    ):
        dm = ChempropDataModule()
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert isinstance(ds, chemprop_data.MoleculeDataset)

    def test_featurize_dataset_length(self, small_mol_list, small_regression_y):
        dm = ChempropDataModule()
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert len(ds) == len(small_mol_list)

    def test_featurize_with_features(self, small_mol_list, small_regression_y):
        dm = ChempropDataModule(feature_list=["estate"])
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        assert isinstance(ds, chemprop_data.MoleculeDataset)
        # Check x_d is populated
        assert ds[0].x_d is not None

    def test_featurize_test_mode(self, small_mol_list, small_regression_y):
        dm = ChempropDataModule()
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        ds_test = dm.featurize(
            small_mol_list[:3], small_regression_y[:3], is_training=False, n_jobs=1
        )
        assert len(ds_test) == 3


# ===================================================================
# Dataloader
# ===================================================================


class TestChempropDataloader:
    def test_create_dataloader(self, small_mol_list, small_regression_y):
        dm = ChempropDataModule(batch_size=4)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, n_jobs=1
        )
        loader = dm.create_dataloader(ds, is_training=True)
        batch = next(iter(loader))
        # Chemprop batches are BatchMolGraph or similar
        assert batch is not None


# ===================================================================
# State dict
# ===================================================================


class TestChempropStateDict:
    def test_state_dict_keys(self, small_mol_list, small_regression_y):
        dm = ChempropDataModule()
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()
        assert "ID" in sd
        assert "use_features" in sd
        assert "y_scaler" in sd

    def test_load_state_dict_roundtrip_no_features(
        self, small_mol_list, small_regression_y
    ):
        dm = ChempropDataModule()
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()

        dm2 = ChempropDataModule.dummy()
        dm2.load_state_dict(sd)
        assert dm2.use_features is False

    def test_load_state_dict_roundtrip_with_features(
        self, small_mol_list, small_regression_y
    ):
        dm = ChempropDataModule(feature_list=["estate"])
        dm.featurize(small_mol_list, small_regression_y, is_training=True, n_jobs=1)
        sd = dm.state_dict()

        dm2 = ChempropDataModule.dummy()
        dm2.load_state_dict(sd)
        assert dm2.use_features is True


# ===================================================================
# Dummy
# ===================================================================


class TestChempropDummy:
    def test_dummy_creation(self):
        dm = ChempropDataModule.dummy()
        assert isinstance(dm, ChempropDataModule)
