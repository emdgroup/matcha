"""Base class for sklearn-compatible chemical language model (CLM) wrappers."""

from matcha.sklearn.base_sklearn_model import (
    BaseScikitLearnModel,
)
from matcha.sklearn.managers import CLMTrainingManager
from matcha.datamodules import (
    CLMDataModule,
    TabularDataModule,
    CombinedDataModule,
)

from rdkit.Chem.rdchem import Mol
from torch.utils.data import StackDataset
import numpy as np
from matcha.datamodules.classic.rdkit_engine import Engine
import torch


class BaseScikitLearnCLM(BaseScikitLearnModel):
    """Base class for all sklearn-compatible CLM models.

    Not meant to be instantiated directly; serves as a parent class for each
    CLM model variant. Adapts :class:`BaseScikitLearnModel` for chemical
    language inputs by configuring CLM-specific datamodules and training
    managers.
    """

    def _adapt_dict_for_modality(self, datamodule_params, model_params):
        """Adapt parameter dictionaries for CLM-specific requirements.

        Computes the additional molecular feature dimensionality from the
        feature list (if provided) and sets a default vocabulary size placeholder.

        :param dict datamodule_params: datamodule configuration dictionary.
        :param dict model_params: model configuration dictionary.
        :returns: the updated (datamodule_params, model_params) tuple.
        :rtype: tuple[dict, dict]
        """
        if datamodule_params["feature_list"] is not None:
            input_dim = Engine().calculate_feature_dim(
                datamodule_params["feature_list"]
            )
            model_params["additional_mol_features_dim"] = input_dim

        # set to default empty vocabulary
        model_params["enc_num_characters"] = 4

        return datamodule_params, model_params

    def _create_datamodule(self, datamodule_params, train_params):
        """Create and configure the CLM datamodule.

        If a ``feature_list`` is provided, wraps a :class:`CLMDataModule` and
        a :class:`TabularDataModule` in a :class:`CombinedDataModule`. Otherwise
        creates a standalone :class:`CLMDataModule`.

        :param dict datamodule_params: datamodule configuration dictionary.
        :param dict train_params: training parameters (must include ``batch_size``).
        """
        datamodule_params = self._parse_label_transform_map(datamodule_params)
        feature_list = datamodule_params.pop("feature_list")
        datamodule_params["num_test_augmentations"] = datamodule_params[
            "num_augmentations"
        ]

        if feature_list is not None:
            self._datamodule_manager.datamodule = CombinedDataModule(
                [
                    CLMDataModule(**datamodule_params),
                    TabularDataModule(feature_list=feature_list),
                ]
            )
        else:
            self._datamodule_manager.datamodule = CLMDataModule(**datamodule_params)

        self.datamodule.params.batch_size = train_params["batch_size"]
        self.datamodule.params.num_workers = 0

    def _create_model(self, model_dict):
        """Overwrites the parent method to add the enc_num_characters parameter
        to the model instance.
        """
        if isinstance(self.datamodule, CLMDataModule):
            model_dict["enc_num_characters"] = self.datamodule.params.num_tokens

        elif isinstance(self.datamodule, CombinedDataModule):
            clm_datamodules = [
                f for f in self.datamodule.datamodules if isinstance(f, CLMDataModule)
            ]
            if len(clm_datamodules) == 0:
                raise ValueError("No CLMDataModule found in the CombinedDataModule")
            elif len(clm_datamodules) > 1:
                raise ValueError(
                    "Multiple CLMDataModules found in the CombinedDataModule"
                )
            else:
                model_dict["enc_num_characters"] = clm_datamodules[0].params.num_tokens

        else:
            raise ValueError("SOMETHING WENT WRONG")

        super()._create_model(model_dict)

    def _start_setup(self):
        """Override to use CLMTrainingManager instead of the default TrainingManager."""
        super()._start_setup()
        self._training_manager = CLMTrainingManager()

    def fit(
        self,
        x: list[Mol] | list[str] | StackDataset,
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        validation_set: StackDataset | None = None,
    ):
        """Runs training with the desired model architecture.

        :param list[Mol] | StackDataset x: either a list of molecules
            or a StackDataset computed by the appropriate datamodule
        :param np.ndarray | None y: property labels in a numpy array or None
        :param list[str] | None bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than'
        :param StackDataset | None validation_set: pre-transformed data for early-stopping
        """
        self.logger.info("Fit: beginning fit")
        self._datamodule_manager.prepare_fit_datasets(
            x=x,
            y=y,
            bound_mask=bound_mask,
            validation_set=validation_set,
            transform_fn=self.transform,
            early_stopping=self._training_manager.params.early_stopping,
            seed=self._training_manager.params.seed,
            batch_size=self._training_manager.params.batch_size,
        )

        self._inner_fit()

    def _inner_predict(
        self,
        x: list[Mol] | list[str] | StackDataset,
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Run prediction with test-time augmentation averaging.

        Calls the parent ``_inner_predict`` to obtain raw predictions, then
        reshapes and averages across SMILES augmentations when
        ``num_test_augmentations > 0``.

        :param x: input molecules as RDKit Mol objects, SMILES strings, or a
            pre-transformed StackDataset.
        :param accelerator: hardware accelerator override (e.g. ``'cpu'``).
        :param devices: number of devices override.
        :param batch_size: batch size override for inference.
        :returns: averaged prediction tensor of shape ``(N, num_endpoints)``.
        :rtype: torch.Tensor
        """
        # Get the number of test augmentations before prediction, handling combined case
        try:
            num_augmentations = self.datamodule.params.num_test_augmentations
        except Exception:
            try:
                num_augmentations = self.datamodule.datamodules[
                    0
                ].params.num_test_augmentations
            except Exception:
                num_augmentations = 0

        augmented_output = super()._inner_predict(x, accelerator, devices, batch_size)

        # If no augmentations, return as is
        if num_augmentations == 0:
            return augmented_output

        # Reshape and average across augmentations
        # The dataset is created with structure: [orig_0, orig_1, ..., aug1_0, aug1_1, ..., aug_k_0, aug_k_1, ...]
        # augmented_output shape: (N * (1 + num_augmentations), num_outputs)
        n_originals = augmented_output.shape[0] // (1 + num_augmentations)

        # Extract original predictions (first N samples)
        original_preds = augmented_output[:n_originals]

        # Extract and stack augmented predictions
        # Each augmentation has n_originals samples
        augmented_preds = []
        for aug_idx in range(num_augmentations):
            start_idx = n_originals * (1 + aug_idx)
            end_idx = start_idx + n_originals
            augmented_preds.append(augmented_output[start_idx:end_idx])

        # Stack all predictions: (num_augmentations + 1, n_originals, num_outputs)
        all_preds = torch.stack([original_preds] + augmented_preds, dim=0)

        # Average across augmentations (dim=0)
        averaged_output = all_preds.mean(dim=0)

        return averaged_output
