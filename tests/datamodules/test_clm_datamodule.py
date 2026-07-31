"""Tests for CLMDataModule."""

from torch.utils.data import StackDataset

from matcha.datamodules.classic.clm_datamodule import (
    CLMDataModule,
    batch_moltosmiles,
    batch_smiles_tokenize,
)
from matcha.datamodules.base_datamodule import DataModuleRegistry


# ===================================================================
# Module-level utilities
# ===================================================================


class TestBatchMolToSmiles:
    def test_returns_list_of_strings(self, small_mol_list):
        result = batch_moltosmiles(small_mol_list)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


class TestBatchSmilesTokenize:
    def test_returns_list_of_token_lists(self):
        smiles = ["CCO", "c1ccccc1"]
        result = batch_smiles_tokenize(smiles)
        assert isinstance(result, list)
        assert all(isinstance(tokens, list) for tokens in result)

    def test_basic_tokenization(self):
        result = batch_smiles_tokenize(["CCO"])
        tokens = result[0]
        assert "C" in tokens
        assert "O" in tokens


# ===================================================================
# Construction
# ===================================================================


class TestCLMDataModuleInit:
    def test_default_construction(self):
        dm = CLMDataModule()
        assert dm.params.datamodule_type == "clm"
        assert dm.params.max_length == 200
        assert dm.params.num_augmentations == 3

    def test_custom_params(self):
        dm = CLMDataModule(max_length=100, num_augmentations=1)
        assert dm.params.max_length == 100
        assert dm.params.num_augmentations == 1

    def test_initial_dictionary(self):
        dm = CLMDataModule()
        assert dm.params.dictionary == {"pad": 0, "unk": 1, "cls": 2, "mask": 3}

    def test_registry_has_clm(self):
        assert "clm" in DataModuleRegistry


# ===================================================================
# Internal methods
# ===================================================================


class TestCLMMolToSmiles:
    def test_canonical(self, small_mol_list):
        dm = CLMDataModule()
        smi = dm._mol_to_smiles(small_mol_list, random=False, n_jobs=1)
        assert len(smi) == len(small_mol_list)
        assert all(isinstance(s, str) for s in smi)

    def test_random(self, small_mol_list):
        dm = CLMDataModule()
        smi = dm._mol_to_smiles(small_mol_list, random=True, n_jobs=1)
        assert len(smi) == len(small_mol_list)


class TestCLMTokenize:
    def test_tokenize_smiles(self):
        dm = CLMDataModule()
        tokens = dm._tokenize_smiles(["CCO", "c1ccccc1"], n_jobs=1)
        assert len(tokens) == 2
        assert all(isinstance(t, list) for t in tokens)


class TestCLMPadTokens:
    def test_pad_to_max_length(self):
        dm = CLMDataModule(max_length=10)
        tokens = [["C", "C", "O"]]
        padded = dm._pad_tokens(tokens)
        assert len(padded[0]) == 10
        assert padded[0][0] == "cls"
        assert padded[0][-1] == "pad"

    def test_truncate_long_sequences(self):
        dm = CLMDataModule(max_length=5)
        tokens = [["C"] * 20]
        padded = dm._pad_tokens(tokens)
        assert len(padded[0]) == 5


class TestCLMDictionary:
    def test_get_dictionary_from_tokens(self):
        dm = CLMDataModule()
        tokens = [["C", "C", "O"], ["N", "C"]]
        dm._get_dictionary(tokens)
        assert "C" in dm._cached_dictionary
        assert "O" in dm._cached_dictionary
        assert "N" in dm._cached_dictionary
        assert "pad" in dm._cached_dictionary
        assert "unk" in dm._cached_dictionary

    def test_encode_tokens(self):
        dm = CLMDataModule()
        dm._cached_dictionary = {"pad": 0, "unk": 1, "cls": 2, "C": 3, "O": 4}
        encoded = dm._encode_tokens([["cls", "C", "O", "pad"]])
        assert encoded == [[2, 3, 4, 0]]

    def test_encode_unknown_token(self):
        dm = CLMDataModule()
        dm._cached_dictionary = {"pad": 0, "unk": 1, "cls": 2}
        encoded = dm._encode_tokens([["cls", "UNKNOWN"]])
        assert encoded[0][1] == 1  # mapped to "unk"


# ===================================================================
# Featurization (regression)
# ===================================================================


class TestCLMFeaturizeRegression:
    def test_featurize_returns_stack_dataset(self, small_mol_list, small_regression_y):
        dm = CLMDataModule(max_length=100, num_augmentations=0)
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        assert isinstance(ds, StackDataset)

    def test_featurize_keys(self, small_mol_list, small_regression_y):
        dm = CLMDataModule(max_length=100, num_augmentations=0)
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        item = ds[0]
        assert "token_ids" in item
        assert "y" in item

    def test_featurize_token_shape(self, small_mol_list, small_regression_y):
        max_len = 100
        dm = CLMDataModule(max_length=max_len, num_augmentations=0)
        ds = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        assert ds.datasets["token_ids"].shape[1] == max_len

    def test_dictionary_populated_after_training(
        self, small_mol_list, small_regression_y
    ):
        dm = CLMDataModule(max_length=100, num_augmentations=0)
        dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        assert len(dm.params.dictionary) > 4  # More than special tokens


class TestCLMFeaturizeWithAugmentation:
    def test_augmentation_increases_dataset_size(
        self, small_mol_list, small_regression_y
    ):
        n = len(small_mol_list)
        dm = CLMDataModule(max_length=100, num_augmentations=2)
        ds = dm.featurize(
            small_mol_list, small_regression_y, is_training=True, augment=True, n_jobs=1
        )
        # canonical + 2 augmentations = 3x size
        assert ds.datasets["token_ids"].shape[0] == n * 3

    def test_no_augmentation_when_not_training(
        self, small_mol_list, small_regression_y
    ):
        dm = CLMDataModule(
            max_length=100,
            num_augmentations=2,
            num_test_augmentations=0,
        )
        dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        ds_test = dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=False,
            augment=True,
            n_jobs=1,
        )
        # With 0 test augmentations, size should be original
        assert ds_test.datasets["token_ids"].shape[0] == len(small_mol_list)


class TestCLMFeaturizeTest:
    def test_featurize_test_uses_fitted_dictionary(
        self, small_mol_list, small_regression_y
    ):
        dm = CLMDataModule(max_length=100, num_augmentations=0)
        dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        ds_test = dm.featurize(
            small_mol_list[:2],
            small_regression_y[:2],
            is_training=False,
            augment=False,
            n_jobs=1,
        )
        assert ds_test.datasets["token_ids"].shape[0] == 2

    def test_featurize_test_without_y(self, small_mol_list, small_regression_y):
        dm = CLMDataModule(max_length=100, num_augmentations=0)
        dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        ds_test = dm.featurize(
            small_mol_list[:2], None, is_training=False, augment=False, n_jobs=1
        )
        assert ds_test.datasets["y"].shape[0] == 2


# ===================================================================
# State dict
# ===================================================================


class TestCLMStateDict:
    def test_state_dict_keys(self, small_mol_list, small_regression_y):
        dm = CLMDataModule(max_length=100, num_augmentations=0)
        dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        sd = dm.state_dict()
        assert "ID" in sd
        assert "dictionary" in sd
        assert "y_scaler" in sd
        assert "params" in sd

    def test_load_state_dict_roundtrip(self, small_mol_list, small_regression_y):
        dm = CLMDataModule(max_length=100, num_augmentations=0)
        dm.featurize(
            small_mol_list,
            small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        sd = dm.state_dict()

        dm2 = CLMDataModule.dummy()
        dm2.load_state_dict(sd)
        assert dm2.params.max_length == 100


# ===================================================================
# Dummy
# ===================================================================


class TestCLMDummy:
    def test_dummy_creation(self):
        dm = CLMDataModule.dummy()
        assert isinstance(dm, CLMDataModule)
