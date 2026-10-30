"""Chemical language model featurization for SMILES-based neural network training."""

from torch import tensor, float32
from torch.utils.data import StackDataset
import re

from rdkit.Chem.rdchem import Mol
import numpy as np
from matcha.datamodules.base_datamodule import BaseDataModule, DataModuleRegistry
from matcha.utils.wrapper import Wrapper
from matcha.utils.schemas.datamodules import CLMDataModuleInputModel
from matcha.utils.wrapper import parallelize

# SMILES tokenization regex pattern (adapted from https://www.arxiv.org/pdf/2409.15370v3)
SMILES_REGEX = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

# due to how joblib works, some rdkit functions require an additional wrapper to
# be properly parallelized
wrapped_MolToSmiles = Wrapper("MolToSmiles", "rdkit.Chem")


def batch_moltosmiles(mol_list: list[Mol], **kwargs):
    return [wrapped_MolToSmiles(x, **kwargs) for x in mol_list]


def batch_smiles_tokenize(smi_list: list[str], pattern: str = SMILES_REGEX):
    """Tokenize SMILES strings using regex pattern."""
    return [re.findall(pattern, smi) for smi in smi_list]


@DataModuleRegistry.register("clm")
class CLMDataModule(BaseDataModule):
    """Chemical language representation featurization class.

    Allows users to convert a list of rdkit molecules and labels into a Tensordataset ready
    to be used for SMILES-based neural network training. Compounds are tokenized
    directly from SMILES strings using regex-based tokenization. Common featurization logic is inherited from
    :class:`BaseFeaturizer`.

    The main purpose of the class is to enable the use of :method:`featurize`.
    Please check out :method:`featurize` for further information on the class' usage.

    :param int max_length: max length for the SMILES token sequence

    :param int n_augmentation: number of augmentations to do when encoding a list of
        molecules

    :param int n_test_augmentations: number of augmentations to do when encoding a list of
        molecules for testing

    :param include_canonical: whether to compute canonical SMILES
    """

    def __init__(
        self,
        max_length: int = 200,
        num_augmentations: int = 3,
        num_test_augmentations: int = 0,
        include_canonical: bool = True,
        is_classification: bool = False,
        scaler_type: str = "standard",
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_params: dict = {},
        batch_size: int = 256,
        num_workers: int = 0,
        augment_resonance: bool = False,
    ):
        self.params = CLMDataModuleInputModel(
            max_length=max_length,
            dictionary={"pad": 0, "unk": 1, "cls": 2, "mask": 3},
            num_tokens=4,
            num_augmentations=num_augmentations,
            num_test_augmentations=num_test_augmentations,
            include_canonical=include_canonical,
            is_classification=is_classification,
            scaler_type=scaler_type,
            clip=clip,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_resonance=augment_resonance,
        )

        super().__init__(
            scaler_type=scaler_type,
            label_encoder_params=label_encoder_params,
            label_transform_params=label_transform_params,
            augment_resonance=augment_resonance,
        )

        # Cache for temporary dictionary during feature generation
        self._cached_dictionary = {}

    def _mol_to_smiles(
        self, mol_list: list[Mol], random: bool = True, n_jobs: int = 4
    ) -> list[str]:
        """Converts a list of rdkit molecules to their SMILES strings.

        :param list[Mol] mol_list: list of rdkit molecules to process

        :param bool random: Whether to use random smiles, defaults to True

        :param int n_jobs: number of cores to use for processing the list, defaults to 4

        :return list[str]: list of SMILES strings
        """

        # make deepcopy to avoid in-place editing
        mol_copy = [Mol(x) for x in mol_list]

        # compute either random or canonical
        if random:
            smi = parallelize(
                batch_moltosmiles, mol_copy, n_jobs, doRandom=True, canonical=False
            )

        else:
            smi = parallelize(batch_moltosmiles, mol_copy, n_jobs, canonical=True)

        return smi

    def _tokenize_smiles(self, smi_list: list[str], n_jobs: int = 4) -> list[list[str]]:
        """Converts a list of SMILES to a list of tokenized SMILES.

        :param list[str] smi_list: list of SMILES to process

        :param int n_jobs: number of cores to use when parallelizing, defaults to 4

        :return list[list[str]]: list of tokenized SMILES lists (nested list)
        """
        # tokenize SMILES using regex
        tokens = parallelize(batch_smiles_tokenize, smi_list, n_jobs)
        return tokens

    def _pad_tokens(self, token_list: list[list[str]]) -> list[list[str]]:
        """Pads each tokenized SMILES sublist in a list to self.params.max_length.

        :param list[list[str]] token_list: list of tokenized SMILES lists to pad

        :return list[list[str]]: list of padded token strings
        """
        padded_tokens = []
        for tokens in token_list:
            # add class token for attention-based CLM
            tokens = ["cls"] + tokens

            # either pad, or cut to max length
            if len(tokens) < self.params.max_length:
                tokens = tokens + ["pad"] * (self.params.max_length - len(tokens))
            else:
                tokens = tokens[: self.params.max_length]
            padded_tokens.append(tokens)
        return padded_tokens

    def _get_dictionary(self, token_list: list[list[str]]):
        """Generates a dictionary pairing each unique SMILES token
        to an integer and stores it in the cached dictionary.

        Generates a dictionary from the training set, including four additional
        characters for padding, unknown tokens, the class token, and the mask token.

        :param list[list[str]] token_list: list of tokenized SMILES lists to generate the dict
            from (e.g. the training set)
        """
        unique_tokens = []
        # collect all unique tokens and discard duplicates (this is definitely
        # suboptimal since it keeps pruning, but for now it will do)
        for tokens in token_list:
            unique_tokens += tokens
            unique_tokens = list(set(unique_tokens))

        # add special tokens to unique tokens
        unique_tokens = ["pad", "unk", "cls", "mask"] + unique_tokens

        # create integer IDs for tokens, create dict and save in cache
        idx = list(range(len(unique_tokens)))
        self._cached_dictionary = dict(zip(unique_tokens, idx))

    def _encode_tokens(self, token_list: list[list[str]]) -> list[list[int]]:
        """Converts a list of tokenized SMILES in a nested list of integers.

        Converts a token sequence in a list of integers, by mapping each
        token to a value according to the cached dictionary (if available) or the params dictionary.

        :param list[list[str]] token_list: list of tokenized SMILES lists

        :return list[list[int]]: nested list of integer embeddings for the token list
        """
        # Use cached dictionary if available, otherwise use params dictionary
        dictionary = (
            self._cached_dictionary
            if self._cached_dictionary != {}
            else self.params.dictionary
        )

        encoded_list = []
        for tokens in token_list:
            encoded = []
            for token in tokens:
                try:
                    t = dictionary[token]
                except Exception:
                    t = dictionary["unk"]
                encoded.append(t)
            encoded_list.append(encoded)
        return encoded_list

    def generate_features(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        augment: bool = True,
        is_training: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generates unscaled features for CLM training.

        This method only handles feature generation without any Y scaling.
        Use featurize() for scaled features.

        :param list[Mol] mol_list: list of N rdkit molecules to process

        :param np.ndarray | None y: array (N, X), where X is the number of classes or
            endpoints or None

        :param list[str] | None bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None

        :param bool augment: whether to compute augmentations

        :param bool is_training: Whether to fit the X and Y scalers on the input,
            or leverage pre-existing ones to only normalize

        :param int | None n_jobs: number of cores to use when featurizing the input,
        if None is passed a reasonable n_jobs value will be guessed from the
        amount of data, defaults to None

        :return StackDataset: Unscaled dataset providing keys `token_ids` and `y`
        """
        # validate inputs without scaling
        mol_list, y, bound_mask, n_jobs = self._validate_input(
            mol_list, y, bound_mask, n_jobs
        )

        # get SMILES and tokenize
        smi = self._mol_to_smiles(
            mol_list, random=not self.params.include_canonical, n_jobs=n_jobs
        )
        tokens = self._tokenize_smiles(smi, n_jobs=n_jobs)

        # get dictionary depending on is_training
        if is_training and self.params.dictionary == {
            "pad": 0,
            "unk": 1,
            "cls": 2,
            "mask": 3,
        }:
            self._get_dictionary(tokens)

        # pad and encode
        tokens = self._pad_tokens(tokens)
        encoded = self._encode_tokens(tokens)

        # if augment is True, repeat procedure self.params.num_augmentations times with
        # random smiles and concatenate
        if augment:
            y_i = y.copy()
            n_augmentations = (
                self.params.num_augmentations
                if is_training
                else self.params.num_test_augmentations
            )
            for _ in range(n_augmentations):
                try:
                    smi_i = self._mol_to_smiles(mol_list, random=True, n_jobs=n_jobs)
                    tokens_i = self._tokenize_smiles(smi_i, n_jobs=n_jobs)
                    tokens_i = self._pad_tokens(tokens_i)
                    encoded_i = self._encode_tokens(tokens_i)
                    encoded += encoded_i
                    y = np.concatenate((y, y_i), axis=0)
                except Exception:
                    smi_i = self._mol_to_smiles(mol_list, random=False, n_jobs=n_jobs)
                    tokens_i = self._tokenize_smiles(smi_i, n_jobs=n_jobs)
                    tokens_i = self._pad_tokens(tokens_i)
                    encoded_i = self._encode_tokens(tokens_i)
                    encoded += encoded_i
                    y = np.concatenate((y, y_i), axis=0)

        tokens_tensor = tensor(encoded)
        y_tensor = tensor(y, dtype=float32)

        return StackDataset(token_ids=tokens_tensor, y=y_tensor)

    def featurize(
        self,
        mol_list: list[Mol],
        y: np.ndarray | None = None,
        bound_mask: list[str] | None = None,
        is_training: bool = True,
        augment: bool = True,
        n_jobs: int | None = None,
    ) -> StackDataset:
        """Generates the features for CLM training.

        Processes a list of N molecules and a numpy array (N,X) encoding the
        labels into a TensorDataset, which can then be further processed for
        CLM neural network training.

        This method chains the unscaled feature generation with
        the necessary Y scaling operations.

        Example usage:

        .. code-block:: python
            CLM = CLMFeaturizer(200, 3)
            train_dataset = CLM.featurize(train_mols, train_y, is_training=True)
            test_dataset = CLM.featurize(test_mols, test_y, is_training=False)

        :param list[Mol] mol_list: list of N rdkit molecules to process

        :param np.ndarray | None y: array (N, X), where X is the number of classes or
            endpoints or None

        :param list[str] | None bound_mask: list of str (N) defining whether the value
            is exact or is 'less than' / 'more than', will be ignored if set
            to None

        :param bool is_training: Whether to fit the X and Y scalers on the input,
            or leverage pre-existing ones to only normalize

        :param bool augment: whether to compute augmentations

        :param int | None n_jobs: number of cores to use when featurizing the input,
        if None is passed a reasonable n_jobs value will be guessed from the
        amount of data, defaults to None

        :return StackDataset: Processed dataset providing keys `token_ids` and `y` ready for MLP-like neural network training
        """

        # Apply augmentation if enabled
        if is_training and self._augment_resonance:
            mol_list, y, bound_mask = self.augment(
                mol_list,
                y,
                bound_mask=bound_mask,
                use_resonance=self._augment_resonance,
                n_jobs=n_jobs,
            )

        # Generate unscaled features
        dataset = self.generate_features(
            mol_list, y, bound_mask, augment, is_training, n_jobs
        )

        # Apply scaling based on is_training flag
        if not self.params.is_classification:
            if is_training:
                self.fit(dataset)

            self.transform(dataset)
        elif self._cached_dictionary:
            self.params.dictionary = self._cached_dictionary.copy()
            self.params.num_tokens = len(self.params.dictionary)
            # Clear the cache after committing
            self._cached_dictionary = {}

        # Handle bound mask and classification transformations (only if not regression)
        self._process_y(dataset, bound_mask)

        return dataset

    def fit(self, dataset: StackDataset) -> None:
        """Fits scalers and other stateful transformations on the dataset.

        For CLMDataModule, this commits the cached dictionary to params.dictionary
        and fits the Y scaler.

        :param StackDataset dataset: dataset to fit transformations on
        """
        # Commit cached dictionary to params if it exists
        if self._cached_dictionary:
            self.params.dictionary = self._cached_dictionary.copy()
            self.params.num_tokens = len(self.params.dictionary)
            # Clear the cache after committing
            self._cached_dictionary = {}

        # Fit Y scaler via parent class
        self._fit_y(dataset)

    def transform(self, dataset: StackDataset) -> StackDataset:
        """Applies fitted transformations to the dataset.

        For CLMDataModule, this applies Y scaling transformations.
        The dictionary should already be fitted in params.dictionary.

        :param StackDataset dataset: dataset to transform
        :return StackDataset: transformed dataset (modified in-place)
        """
        # Apply Y scaling if fitted
        self._transform_y(dataset)

        return dataset

    def state_dict(self) -> dict:
        """Utility for MLFlow logging"""

        return {
            "ID": "clm",
            "params": self.params.model_dump(),
            "dictionary": self.params.dictionary,
            "y_scaler": self._y_scaler,
            "label_encoder": self._label_encoder,
            "label_transform": self._label_transform,
        }

    def load_state_dict(self, state_dict: dict):
        """Utility for MLFlow logging"""

        super().load_state_dict(state_dict)

        self.params = CLMDataModuleInputModel(**state_dict["params"])

    @classmethod
    def dummy(cls):
        """Utility to make a dummy class with default params. Can be
        combined with load_state_dict to recreate a datamodule from
        a state dict
        """
        return cls()
