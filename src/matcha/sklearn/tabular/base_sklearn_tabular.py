"""Base class for sklearn-compatible tabular model wrappers."""

from matcha.sklearn.base_sklearn_model import (
    BaseScikitLearnModel,
)
from matcha.datamodules import TabularDataModule
from matcha.datamodules.classic.rdkit_engine import Engine


class BaseScikitLearnTabular(BaseScikitLearnModel):
    """Base class for all sklearn-compatible tabular models.

    Not meant to be instantiated directly; serves as a parent class for
    tabular model variants. Configures a :class:`TabularDataModule` and
    automatically computes the input dimensionality from the specified
    feature list.
    """

    def __init__(self, arg_dict: dict):
        super().__init__(arg_dict)

    def _adapt_dict_for_modality(self, datamodule_params, model_params):
        """Compute and set the input feature dimensionality for tabular models.

        :param dict datamodule_params: datamodule configuration dictionary.
        :param dict model_params: model configuration dictionary.
        :returns: the updated (datamodule_params, model_params) tuple.
        :rtype: tuple[dict, dict]
        """
        input_dim = Engine().calculate_feature_dim(datamodule_params["feature_list"])
        model_params["additional_mol_features_dim"] = input_dim
        return datamodule_params, model_params

    def _create_datamodule(self, datamodule_params, train_params):
        """Create and configure the tabular datamodule.

        :param dict datamodule_params: datamodule configuration dictionary.
        :param dict train_params: training parameters (must include ``batch_size``).
        """
        datamodule_params = self._parse_label_transform_map(datamodule_params)
        self._datamodule_manager.datamodule = TabularDataModule(**datamodule_params)
        self.datamodule.params.batch_size = train_params["batch_size"]
        self.datamodule.params.num_workers = 0
