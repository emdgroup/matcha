"""Base class for sklearn-compatible 3D graph neural network models."""

from matcha.sklearn.graph.base_sklearn_gnn import BaseScikitLearnGNN
from matcha.datamodules import CombinedDataModule, Graph3DDataModule, TabularDataModule


class BaseScikitLearnGNN3D(BaseScikitLearnGNN):
    """Base class for all sklearn API 3D graph models.

    Extends :class:`~matcha.sklearn.graph.BaseScikitLearnGNN` with 3D-specific
    data module creation that uses :class:`~matcha.datamodules.Graph3DDataModule`
    for molecular conformer inputs. Placeholder for potential future custom
    functionalities required for conformers.
    """

    def _create_datamodule(self, datamodule_params, train_params):
        """Create the appropriate data module for 3D graph inputs.

        If ``feature_list`` is provided, creates a
        :class:`~matcha.datamodules.CombinedDataModule` combining a
        :class:`~matcha.datamodules.Graph3DDataModule` with a
        :class:`~matcha.datamodules.TabularDataModule`. Otherwise, creates
        a standalone :class:`~matcha.datamodules.Graph3DDataModule`.

        :param dict datamodule_params: Parameters for the data module, including
            ``feature_list`` (popped before forwarding to the data module constructor).
        :param dict train_params: Training parameters; ``batch_size`` is extracted
            and set on the resulting data module.
        """
        datamodule_params = self._parse_label_transform_map(datamodule_params)
        feature_list = datamodule_params["feature_list"]
        datamodule_params.pop("feature_list")
        if feature_list is not None:
            graph_feat = Graph3DDataModule(**datamodule_params)
            tab_feat = TabularDataModule(feature_list=feature_list)
            self._datamodule_manager.datamodule = CombinedDataModule(
                [graph_feat, tab_feat]
            )
        else:
            self._datamodule_manager.datamodule = Graph3DDataModule(**datamodule_params)
        self.datamodule.params.batch_size = train_params["batch_size"]
        self.datamodule.params.num_workers = 0
