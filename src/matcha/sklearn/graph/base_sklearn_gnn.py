"""Base class for sklearn-compatible graph neural network models."""

from torch.utils.data import DataLoader

from matcha.datamodules import CombinedDataModule, GraphDataModule, TabularDataModule
from matcha.datamodules.classic.graph_datamodule import ATOM_FEAT_DIM, BOND_FEAT_DIM
from matcha.sklearn.base_sklearn_model import (
    BaseScikitLearnModel,
)

from matcha.datamodules.classic.rdkit_engine import Engine


class BaseScikitLearnGNN(BaseScikitLearnModel):
    """Base class for all sklearn-compatible graph neural network models.

    Not meant to be instantiated directly. Subclasses inherit graph-specific
    datamodule creation, modality adaptation (positional encodings, virtual
    nodes, additional molecular features), and a collate-aware dataloader.
    """

    def _adapt_dict_for_modality(self, datamodule_params, model_params):
        """Inject graph-specific dimensions into the model parameter dict.

        Computes additional molecular feature dimensions from the feature list,
        copies positional encoding settings (Laplacian, RWSE, electrostatic,
        distance matrix, RRWP), hard-codes atom/bond input dimensions from
        datamodule constants, and adjusts for virtual node indicator features.

        :param dict datamodule_params: datamodule configuration dictionary.
        :param dict model_params: model configuration dictionary (modified in place).
        :returns: the (datamodule_params, model_params) tuple with model_params updated.
        :rtype: tuple[dict, dict]
        """
        if datamodule_params["feature_list"] is not None:
            input_dim = Engine().calculate_feature_dim(
                datamodule_params["feature_list"]
            )
            model_params["additional_mol_features_dim"] = input_dim

        model_params["enc_laplacian_k"] = datamodule_params["laplacian_k"]
        model_params["enc_rwse_k"] = datamodule_params["rwse_k"]
        model_params["enc_elstatic_k"] = datamodule_params["elstatic_k"]
        model_params["enc_distmat_k"] = datamodule_params["distmat_k"]
        model_params["enc_rrwp_k"] = datamodule_params["rrwp_k"]

        # Hardcode atom/bond input dims from the datamodule constants
        model_params["enc_atom_input_dim"] = ATOM_FEAT_DIM
        model_params["enc_bond_input_dim"] = BOND_FEAT_DIM

        # Adjust for virtual node indicator feature (+1 dimension)
        if datamodule_params["num_virtual_nodes"] > 0:
            model_params["enc_atom_input_dim"] += 1
            model_params["enc_bond_input_dim"] += 1

        return datamodule_params, model_params

    def _create_datamodule(self, datamodule_params, train_params):
        """Create and configure the graph datamodule.

        If a ``feature_list`` is provided, creates a
        :class:`~matcha.datamodules.CombinedDataModule` that pairs a
        :class:`~matcha.datamodules.GraphDataModule` with a
        :class:`~matcha.datamodules.TabularDataModule` for the extra features.
        Otherwise creates a standalone ``GraphDataModule``.

        :param dict datamodule_params: datamodule configuration dictionary.
        :param dict train_params: training parameters containing ``batch_size``.
        """
        datamodule_params = self._parse_label_transform_map(datamodule_params)
        feature_list = datamodule_params["feature_list"]
        datamodule_params.pop("feature_list")
        if feature_list is not None:
            graph_feat = GraphDataModule(**datamodule_params)
            tab_feat = TabularDataModule(feature_list=feature_list)
            self._datamodule_manager.datamodule = CombinedDataModule(
                [graph_feat, tab_feat]
            )
        else:
            self._datamodule_manager.datamodule = GraphDataModule(**datamodule_params)

        self.datamodule.params.batch_size = train_params["batch_size"]
        self.datamodule.params.num_workers = 0

    def _make_dataloader(self, dataset: tuple, shuffle: bool):
        """Adapts the self._make_dataloader method to pass the collate function
        to the dataloader.
        -----------------------

        :param tuple dataset: dataset to batch

        :param bool shuffle: whether to shuffle (important to distinguish between
            train and test sets)

        :return DataLoader: batched dataset ready to be used for train/inference,
            with the appropriate collate function
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.params.training.batch_size,
            collate_fn=self.datamodule.collate_fn,
            shuffle=shuffle,
        )
