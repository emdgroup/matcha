"""CLM Masked Language Modeling DataModule for self-supervised pretraining."""

import torch
from torch import long
from torch.utils.data import StackDataset
import numpy as np

from rdkit.Chem.rdchem import Mol
from matcha.datamodules.classic.clm_datamodule import CLMDataModule
from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.utils.schemas.datamodules import CLMMLMDataModuleInputModel


@DataModuleRegistry.register("clm_mlm")
class CLMMLMDataModule(CLMDataModule):
    """CLM DataModule variant for Masked Language Modeling (MLM) pretraining.

    This datamodule extends CLMDataModule to produce training data suitable for
    self-supervised masked language modeling. Instead of returning labels (y),
    it returns:
    - `token_ids`: The masked input sequence (with some tokens replaced by mask token)
    - `y`: The original token sequence (targets for MLM prediction)
    - `mask`: Boolean mask indicating which positions were masked

    Masking is applied AFTER augmentation to ensure consistency. Special tokens
    (pad, unk, cls, mask) are never masked.

    :param float mask_rate: Fraction of tokens to mask for MLM training, defaults to 0.15

    All other parameters are inherited from CLMDataModule.
    """

    # Special tokens that should never be masked
    _special_token_names = {"pad", "unk", "cls", "mask"}

    def __init__(self, mask_rate: float = 0.15, **kwargs):
        """Initialise the MLM datamodule.

        :param mask_rate: fraction of non-special tokens to mask, defaults to 0.15
        :param kwargs: additional keyword arguments forwarded to
            :class:`CLMDataModule`
        """
        # Set defaults suitable for MLM (no label processing)
        kwargs.setdefault("is_classification", True)  # Avoids Y scaling
        kwargs.setdefault("clip", False)

        super().__init__(**kwargs)

        # Override params with MLM-specific schema
        self.params = CLMMLMDataModuleInputModel(
            mask_rate=mask_rate,
            **{
                k: v
                for k, v in self.params.model_dump().items()
                if k not in ("datamodule_type", "mask_rate")
            },
        )

    def export_to_classic(self) -> CLMDataModule:
        """Return a :class:`CLMDataModule` that mirrors the current state.

        The exported instance inherits the dictionary, max_length, and all
        other CLM-specific settings so that it can be used directly for
        downstream (non-MLM) training or inference.

        :return CLMDataModule: a classic CLM datamodule with the same state
        """
        # Extract CLM-compatible params from current params
        p = self.params
        dm = CLMDataModule(
            max_length=p.max_length,
            num_augmentations=p.num_augmentations,
            num_test_augmentations=p.num_test_augmentations,
            include_canonical=p.include_canonical,
            is_classification=p.is_classification,
            scaler_type=p.scaler_type,
            clip=p.clip,
            label_encoder_params=p.label_encoder_params,
            label_transform_params=p.label_transform_params,
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            augment_resonance=p.augment_resonance,
        )
        # Transfer the learned dictionary
        dm.params.dictionary = p.dictionary.copy()
        dm.params.num_tokens = p.num_tokens
        return dm

    def _get_special_token_ids(self) -> set[int]:
        """Get the set of token IDs that should not be masked.

        :return: set of integer token IDs for pad, unk, cls, and mask tokens
        """
        dictionary = self._cached_dictionary or self.params.dictionary
        return {
            dictionary[name] for name in self._special_token_names if name in dictionary
        }

    def _apply_masking(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply MLM masking to token sequences.

        Randomly replaces a fraction of non-special tokens with the mask token.
        Special tokens (pad, unk, cls, mask) are never masked.

        :param token_ids: original token IDs ``[batch_size, seq_length]``
        :return: tuple of ``(masked_ids, original_ids, mask)`` where *mask* is
            a boolean tensor indicating which positions were masked
        """
        special_ids = self._get_special_token_ids()
        dictionary = self._cached_dictionary or self.params.dictionary
        mask_token_id = dictionary["mask"]

        # Create mask for valid positions (non-special tokens)
        valid_mask = torch.ones_like(token_ids, dtype=torch.bool)
        for special_id in special_ids:
            valid_mask &= token_ids != special_id

        # Random mask based on mask_rate, applied only to valid positions
        random_mask = (
            torch.rand_like(token_ids, dtype=torch.float) < self.params.mask_rate
        )
        final_mask = valid_mask & random_mask

        # Apply masking
        masked_token_ids = token_ids.clone()
        masked_token_ids[final_mask] = mask_token_id

        return masked_token_ids, token_ids.clone(), final_mask

    def generate_features(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        augment: bool = True,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generate tokenized features with MLM masking applied.

        Extends the parent to apply masking after tokenization and augmentation.
        The *y* parameter is ignored; original tokens become the targets.

        :param mol_list: list of RDKit molecules
        :param y: ignored (MLM is self-supervised); a dummy array is used
        :param bound_mask: ignored for MLM
        :param augment: whether to apply SMILES augmentation
        :param is_training: whether to update the dictionary
        :param n_jobs: number of parallel workers
        :return: StackDataset with keys ``token_ids``, ``y``, and ``mask``
        """
        # Create dummy y for parent validation
        if y is None:
            y = np.zeros((len(mol_list), 1), dtype=np.float32)

        # Use parent's generate_features for tokenization + augmentation
        parent_dataset = super().generate_features(
            mol_list, y, bound_mask, augment, is_training, n_jobs
        )

        # Apply MLM masking (post-augmentation)
        token_ids = parent_dataset.datasets["token_ids"]
        masked_ids, original_ids, mask = self._apply_masking(token_ids)

        return StackDataset(
            token_ids=masked_ids,
            y=original_ids.to(dtype=long),
            mask=mask,
        )

    def featurize(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        augment: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generate a dataset ready for MLM pretraining.

        Optionally applies resonance augmentation, then delegates to
        :meth:`generate_features` for tokenization and masking. Commits
        the dictionary on the first training call.

        :param mol_list: list of RDKit molecules
        :param y: ignored (MLM is self-supervised)
        :param bound_mask: ignored for MLM
        :param is_training: whether to commit the dictionary after featurization
        :param augment: whether to apply SMILES augmentation
        :param n_jobs: number of parallel workers
        :return: StackDataset with keys ``token_ids``, ``y``, and ``mask``
        """
        # Apply resonance augmentation if enabled
        if is_training and self._augment_resonance:
            mol_list, _, _ = self.augment(
                mol_list,
                y,
                bound_mask=bound_mask,
                use_resonance=self._augment_resonance,
                n_jobs=n_jobs,
            )

        dataset = self.generate_features(
            mol_list, y, bound_mask, augment, is_training, n_jobs
        )

        # Commit dictionary if training
        if is_training and self._cached_dictionary:
            self.params.dictionary = self._cached_dictionary.copy()
            self.params.num_tokens = len(self.params.dictionary)
            self._cached_dictionary = {}

        return dataset

    def fit(self, dataset: StackDataset) -> None:
        """Commit the cached dictionary. No Y scaling for MLM.

        :param dataset: the featurized StackDataset (unused beyond triggering commit)
        """
        if self._cached_dictionary:
            self.params.dictionary = self._cached_dictionary.copy()
            self.params.num_tokens = len(self.params.dictionary)
            self._cached_dictionary = {}

    def transform(self, dataset: StackDataset) -> StackDataset:
        """No-op for MLM since we don't scale Y.

        :param dataset: featurized StackDataset
        :return: the dataset unchanged
        """
        return dataset

    def state_dict(self) -> dict:
        """Serialise state for MLFlow logging.

        :return: dict containing ID, params, and dictionary
        """
        return {
            "ID": "clm_mlm",
            "params": self.params.model_dump(),
            "dictionary": self.params.dictionary,
        }

    def load_state_dict(self, state_dict: dict):
        """Restore state from a previously serialised dict.

        :param state_dict: dict produced by :meth:`state_dict`
        """
        self.params = CLMMLMDataModuleInputModel(**state_dict["params"])

    @classmethod
    def dummy(cls):
        """Create a dummy instance with default parameters.

        :return: a new :class:`CLMMLMDataModule` with default settings
        """
        return cls()
