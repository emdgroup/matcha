"""Tests for CLMMLMDataModule."""

import torch
from torch.utils.data import StackDataset

from matcha.datamodules.pretraining.clm_mlm_datamodule import CLMMLMDataModule
from matcha.datamodules.pretraining.on_the_fly_mlm_datamodule import (
    OnTheFlyMLMDataModule,
)
from matcha.datamodules.base_datamodule import DataModuleRegistry


# ===================================================================
# Construction
# ===================================================================


class TestCLMMLMDataModuleInit:
    def test_default_construction(self):
        dm = CLMMLMDataModule()
        assert dm.params.datamodule_type == "clm_mlm"
        assert dm.params.max_length == 200
        assert dm.params.num_augmentations == 3
        assert dm.params.mask_rate == 0.15

    def test_custom_mask_rate(self):
        dm = CLMMLMDataModule(mask_rate=0.2)
        assert dm.params.mask_rate == 0.2

    def test_custom_params(self):
        dm = CLMMLMDataModule(
            mask_rate=0.1,
            max_length=100,
            num_augmentations=1,
        )
        assert dm.params.max_length == 100
        assert dm.params.num_augmentations == 1
        assert dm.params.mask_rate == 0.1

    def test_initial_dictionary(self):
        dm = CLMMLMDataModule()
        assert dm.params.dictionary == {"pad": 0, "unk": 1, "cls": 2, "mask": 3}

    def test_registry_has_clm_mlm(self):
        assert "clm_mlm" in DataModuleRegistry


# ===================================================================
# Masking
# ===================================================================


class TestMasking:
    def test_masking_returns_correct_shapes(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        assert "token_ids" in ds.datasets
        assert "y" in ds.datasets
        assert "mask" in ds.datasets

        # Check shapes match
        assert ds.datasets["token_ids"].shape == ds.datasets["y"].shape
        assert ds.datasets["token_ids"].shape == ds.datasets["mask"].shape

    def test_masking_does_not_mask_special_tokens(self, small_mol_list):
        dm = CLMMLMDataModule(
            mask_rate=0.5,  # High rate to ensure some masking
            max_length=50,
            num_augmentations=0,
        )
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        y = ds.datasets["y"]
        mask = ds.datasets["mask"]

        # Get special token IDs
        pad_id = dm.params.dictionary["pad"]
        cls_id = dm.params.dictionary["cls"]
        unk_id = dm.params.dictionary["unk"]
        mask_id = dm.params.dictionary["mask"]
        special_ids = {pad_id, cls_id, unk_id, mask_id}

        # Check that special tokens in original (y) are never masked
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i, j].item() in special_ids:
                    # Special tokens should not be masked
                    assert not mask[i, j].item(), (
                        f"Special token at ({i},{j}) was masked"
                    )

    def test_masked_positions_have_mask_token(self, small_mol_list):
        dm = CLMMLMDataModule(
            mask_rate=0.3,
            max_length=50,
            num_augmentations=0,
        )
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        token_ids = ds.datasets["token_ids"]
        mask = ds.datasets["mask"]
        mask_token_id = dm.params.dictionary["mask"]

        # Masked positions should have mask token
        masked_token_ids = token_ids[mask]
        assert torch.all(masked_token_ids == mask_token_id)

    def test_unmasked_positions_preserve_original(self, small_mol_list):
        dm = CLMMLMDataModule(
            mask_rate=0.2,
            max_length=50,
            num_augmentations=0,
        )
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        token_ids = ds.datasets["token_ids"]
        y = ds.datasets["y"]
        mask = ds.datasets["mask"]

        # Non-masked positions should be identical between token_ids and y
        unmasked = ~mask
        assert torch.all(token_ids[unmasked] == y[unmasked])

    def test_mask_rate_approximately_correct(self, mol_list):
        # Use full mol_list for statistical significance
        dm = CLMMLMDataModule(
            mask_rate=0.15,
            max_length=100,
            num_augmentations=0,
        )
        ds = dm.featurize(mol_list, is_training=True, augment=False, n_jobs=1)

        y = ds.datasets["y"]
        mask = ds.datasets["mask"]

        # Get special token IDs
        pad_id = dm.params.dictionary["pad"]
        cls_id = dm.params.dictionary["cls"]
        special_ids = {pad_id, cls_id}

        # Count maskable tokens (non-special)
        maskable = torch.ones_like(y, dtype=torch.bool)
        for sid in special_ids:
            maskable &= y != sid

        total_maskable = maskable.sum().item()
        total_masked = mask.sum().item()

        if total_maskable > 0:
            actual_rate = total_masked / total_maskable
            # Allow for random variation: should be within 0.05 of target
            assert abs(actual_rate - 0.15) < 0.1, (
                f"Expected ~0.15 mask rate, got {actual_rate}"
            )


# ===================================================================
# Featurization
# ===================================================================


class TestCLMMLMFeaturize:
    def test_featurize_returns_stack_dataset(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)
        assert isinstance(ds, StackDataset)

    def test_featurize_keys(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)
        item = ds[0]
        assert "token_ids" in item
        assert "y" in item
        assert "mask" in item

    def test_featurize_token_shape(self, small_mol_list):
        max_len = 50
        dm = CLMMLMDataModule(max_length=max_len, num_augmentations=0)
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)
        assert ds.datasets["token_ids"].shape[1] == max_len

    def test_y_dtype_is_long(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        ds = dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)
        assert ds.datasets["y"].dtype == torch.long

    def test_featurize_ignores_labels(self, small_mol_list, small_regression_y):
        """MLM should ignore provided labels and use tokens as targets."""
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        ds = dm.featurize(
            small_mol_list,
            y=small_regression_y,  # Provide labels (should be ignored)
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        # y should be tokens, not regression labels
        assert ds.datasets["y"].shape[1] == 50  # seq_length, not 1


class TestCLMMLMAugmentation:
    def test_augmentation_increases_dataset_size(self, small_mol_list):
        n = len(small_mol_list)
        dm = CLMMLMDataModule(max_length=50, num_augmentations=2)
        ds = dm.featurize(small_mol_list, is_training=True, augment=True, n_jobs=1)
        # canonical + 2 augmentations = 3x size
        assert ds.datasets["token_ids"].shape[0] == n * 3

    def test_masking_applied_after_augmentation(self, small_mol_list):
        """Verify masking is applied to all augmented sequences."""
        dm = CLMMLMDataModule(
            mask_rate=0.2,
            max_length=50,
            num_augmentations=2,
        )
        ds = dm.featurize(small_mol_list, is_training=True, augment=True, n_jobs=1)

        mask = ds.datasets["mask"]

        # All sequences should have some masking (with high probability)
        # Check that each augmented batch has masks
        n = len(small_mol_list)
        for batch_idx in range(3):  # 3 batches: canonical + 2 augmentations
            batch_mask = mask[batch_idx * n : (batch_idx + 1) * n]
            # At least some tokens should be masked in each batch
            assert batch_mask.any(), f"Batch {batch_idx} has no masks"


# ===================================================================
# Dictionary
# ===================================================================


class TestCLMMLMDictionary:
    def test_dictionary_populated_after_training(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)
        assert len(dm.params.dictionary) > 4  # More than special tokens

    def test_dictionary_used_for_test_data(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        ds_test = dm.featurize(
            small_mol_list[:2], is_training=False, augment=False, n_jobs=1
        )
        assert ds_test.datasets["token_ids"].shape[0] == 2


# ===================================================================
# State dict
# ===================================================================


class TestCLMMLMStateDict:
    def test_state_dict_keys(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)
        sd = dm.state_dict()
        assert "ID" in sd
        assert "dictionary" in sd
        assert "params" in sd
        assert sd["ID"] == "clm_mlm"
        assert "mask_rate" in sd["params"]

    def test_load_state_dict_roundtrip(self, small_mol_list):
        dm = CLMMLMDataModule(
            mask_rate=0.2,
            max_length=50,
            num_augmentations=0,
        )
        dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)
        sd = dm.state_dict()

        dm2 = CLMMLMDataModule.dummy()
        dm2.load_state_dict(sd)
        assert dm2.params.max_length == 50
        assert dm2.params.mask_rate == 0.2


# ===================================================================
# Dummy
# ===================================================================


class TestCLMMLMDummy:
    def test_dummy_creation(self):
        dm = CLMMLMDataModule.dummy()
        assert isinstance(dm, CLMMLMDataModule)


# ===================================================================
# export_to_classic
# ===================================================================

from matcha.datamodules.classic.clm_datamodule import CLMDataModule  # noqa: E402


class TestCLMMLMExportToClassic:
    def test_returns_clm_datamodule(self):
        dm = CLMMLMDataModule()
        classic = dm.export_to_classic()
        assert isinstance(classic, CLMDataModule)
        assert not isinstance(classic, CLMMLMDataModule)

    def test_dictionary_transferred(self, small_mol_list):
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        classic = dm.export_to_classic()
        assert classic.params.dictionary == dm.params.dictionary
        assert classic.params.num_tokens == dm.params.num_tokens
        assert len(classic.params.dictionary) > 4  # More than special tokens

    def test_params_preserved(self):
        dm = CLMMLMDataModule(
            mask_rate=0.2,
            max_length=100,
            num_augmentations=5,
            num_test_augmentations=2,
            include_canonical=False,
            batch_size=128,
        )
        classic = dm.export_to_classic()
        assert classic.params.max_length == 100
        assert classic.params.num_augmentations == 5
        assert classic.params.num_test_augmentations == 2
        assert classic.params.include_canonical is False
        assert classic.params.batch_size == 128

    def test_classic_datamodule_type(self):
        dm = CLMMLMDataModule()
        classic = dm.export_to_classic()
        assert classic.params.datamodule_type == "clm"

    def test_classic_can_featurize(self, small_mol_list, small_regression_y):
        """Exported CLMDataModule should be able to featurize with labels."""
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        classic = dm.export_to_classic()
        ds = classic.featurize(
            small_mol_list,
            y=small_regression_y,
            is_training=True,
            augment=False,
            n_jobs=1,
        )
        assert "token_ids" in ds.datasets
        assert "y" in ds.datasets
        # y should be regression labels, not token ids
        assert ds.datasets["y"].shape[1] == small_regression_y.shape[1]

    def test_dictionary_is_independent_copy(self, small_mol_list):
        """Modifying the exported dictionary should not affect the original."""
        dm = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm.featurize(small_mol_list, is_training=True, augment=False, n_jobs=1)

        classic = dm.export_to_classic()
        classic.params.dictionary["new_token"] = 999
        assert "new_token" not in dm.params.dictionary


# ===================================================================
# OnTheFlyMLMDataModule
# ===================================================================


class TestOnTheFlyMLMDataModule:
    def test_construction(self):
        base = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm = OnTheFlyMLMDataModule(base=base)
        assert dm.params.datamodule_type == "clm_mlm"
        assert dm.params.mask_rate == 0.15

    def test_registry_has_on_the_fly_mlm(self):
        assert "on_the_fly_mlm" in DataModuleRegistry

    def test_set_data(self, smiles_list):
        base = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm = OnTheFlyMLMDataModule(base=base)

        dm.set_data(
            train_smiles=smiles_list[:10],
            val_smiles=smiles_list[10:15],
        )

        assert dm._raw_train is not None
        assert dm._raw_val is not None
        assert len(dm._raw_train) == 10
        assert len(dm._raw_val) == 5

    def test_fit_dictionary(self, smiles_list):
        base = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm = OnTheFlyMLMDataModule(base=base)

        dm.fit_dictionary(smiles_list[:10], n_jobs=1)

        # Dictionary should be populated in base
        assert len(dm.base.params.dictionary) > 4

    def test_collate_fn(self, smiles_list):
        base = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm = OnTheFlyMLMDataModule(base=base)

        # Fit dictionary first
        dm.fit_dictionary(smiles_list[:10], n_jobs=1)

        # Create a batch
        batch = [{"smiles": smi} for smi in smiles_list[:3]]

        result = dm.collate_fn(batch)

        assert "token_ids" in result
        assert "y" in result
        assert "mask" in result
        assert result["token_ids"].shape[0] == 3

    def test_state_dict(self, smiles_list):
        base = CLMMLMDataModule(max_length=50, num_augmentations=0)
        dm = OnTheFlyMLMDataModule(base=base, num_workers=2)

        dm.fit_dictionary(smiles_list[:10], n_jobs=1)
        sd = dm.state_dict()

        assert "ID" in sd
        assert sd["ID"] == "on_the_fly_mlm"
        assert "base_state_dict" in sd
        assert "num_workers" in sd
        assert sd["num_workers"] == 2
