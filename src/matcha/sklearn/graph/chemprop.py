"""Sklearn-compatible Chemprop (D-MPNN) classifiers and regressors."""

from matcha.sklearn.base_sklearn_model import (
    ScikitLearnRegressorMixin,
    ScikitLearnClassifierMixin,
    BaseScikitLearnModel,
)
from matcha.torch.models.classic import ChempropModel
from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry

from matcha.datamodules.classic.rdkit_engine import Engine
import numpy as np
from rdkit.Chem.rdchem import Mol
from matcha.datamodules import ChempropDataModule
from matcha.utils import silence_nuisance_warnings


class ChempropMixin:
    """Mixin providing Chemprop-specific datamodule creation and modality adaptation.

    Overrides the default graph datamodule logic from
    :class:`~matcha.sklearn.graph.BaseScikitLearnGNN` to use
    :class:`~matcha.datamodules.ChempropDataModule` instead, and computes
    additional molecular feature dimensions from the feature list.
    """

    def _adapt_dict_for_modality(self, datamodule_params, model_params):
        """Compute additional molecular feature dimensions from the feature list.

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
        return datamodule_params, model_params

    def _create_datamodule(self, datamodule_params, train_params):
        """Create and configure a :class:`~matcha.datamodules.ChempropDataModule`.

        :param dict datamodule_params: datamodule configuration dictionary.
        :param dict train_params: training parameters containing ``batch_size``.
        """
        datamodule_params = self._parse_label_transform_map(datamodule_params)
        self._datamodule_manager.datamodule = ChempropDataModule(**datamodule_params)
        self.datamodule.params.batch_size = train_params["batch_size"]
        self.datamodule.params.num_workers = 0


@ScikitLearnModelRegistry.register()
class ChempropClassifier(
    BaseScikitLearnModel, ScikitLearnClassifierMixin, ChempropMixin
):
    """Chemprop (Directed Message Passing Neural Network) classifier.

    Uses a directed message-passing neural network (D-MPNN) that operates on
    molecular graphs for property prediction. Compatible with classification
    datasets only.

    Inherits from :class:`~matcha.sklearn.base_sklearn_model.BaseScikitLearnModel`
    and :class:`ChempropMixin` for Chemprop-specific datamodule handling.

    References:

    - Yang et al., *Analyzing Learned Molecular Representations for Property Prediction*
      https://arxiv.org/abs/1904.01561

    Example usage:

    .. code-block:: python

        model = ChempropClassifier()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_num_layers: number of message-passing layers, defaults to 3
    :param int enc_atom_hidden_dim: hidden dimensionality in the encoder,
        defaults to 300
    :param str enc_readout: encoder readout aggregation, defaults to 'norm'
    :param float enc_dropout: dropout rate in the encoder, defaults to 0.1
    :param str enc_activation: activation function in the encoder, defaults to 'relu'
    :param int pred_hidden_dim: hidden dimensionality in the predictor FFN,
        defaults to 300
    :param int pred_num_layers: number of FFN layers in the predictor, defaults to 2
    :param str pred_activation: activation function in the predictor, defaults to 'relu'
    :param float pred_dropout: dropout rate in the predictor, defaults to 0.1
    :param int num_endpoints: number of endpoints or classes to predict, defaults to 1
    :param str loss_fn: loss function to optimize, defaults to 'bce'
    :param str optimizer: optimizer to use, defaults to 'chemprop'
    :param dict optimizer_args: optimizer arguments, defaults to {'lr': 1e-4}
    :param str scheduler: learning rate scheduler, defaults to 'chemprop'
    :param dict scheduler_args: scheduler arguments, defaults to
        {'warmup_epochs': 2, 'max_lr': 1e-3, 'final_lr': 1e-5}
    :param int num_epochs: number of training epochs, defaults to 20
    :param int batch_size: batch size for training and prediction, defaults to 64
    :param bool stochastic_weight_averaging: whether to use SWA, defaults to False
    :param bool early_stopping: whether to use early stopping, defaults to True
    :param int patience: epochs to wait before early stopping, defaults to 10
    :param int devices: number of devices for training, defaults to 1
    :param str accelerator: hardware accelerator, defaults to 'gpu'
    :param list[str] | None feature_list: molecular feature set to compute,
        defaults to ['rdkit_all_descriptors']
    :param dict label_encoder_params: label encoder parameters, defaults to {}
    :param str | list[str] | dict | None label_transform_map: label transform
        specification, defaults to None
    :param int seed: random seed, defaults to 0
    """

    def __init__(
        self,
        enc_num_layers: int = 3,
        enc_atom_hidden_dim: int = 300,
        enc_readout: str = "norm",
        enc_dropout: float = 0.1,
        enc_activation: str = "relu",
        pred_hidden_dim: int = 300,
        pred_num_layers: int = 2,
        pred_activation: str = "relu",
        pred_dropout: float = 0.1,
        num_endpoints: int = 1,
        loss_fn: str = "bce",
        optimizer: str = "chemprop",
        optimizer_args: dict = {"lr": 1e-4},
        scheduler: str = "chemprop",
        scheduler_args: dict = {"warmup_epochs": 2, "max_lr": 1e-3, "final_lr": 1e-5},
        num_epochs: int = 20,
        batch_size: int = 64,
        stochastic_weight_averaging: bool = False,
        early_stopping: bool = True,
        patience: int = 10,
        devices: int = 1,
        accelerator: str = "gpu",
        feature_list: list[str] | None = ["rdkit_all_descriptors"],
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        seed: int = 0,
    ):
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        self._architecture = ChempropModel
        super(ChempropClassifier, self).__init__(params)
        silence_nuisance_warnings()

    def _adapt_dict_for_modality(self, datamodule_params, model_params):
        return ChempropMixin._adapt_dict_for_modality(
            self, datamodule_params, model_params
        )

    def _create_datamodule(self, datamodule_params, train_params):
        return ChempropMixin._create_datamodule(self, datamodule_params, train_params)

    def predict_proba(
        self,
        x: list[Mol],
        accelerator: str | None = None,
        devices: int | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Allows to return probabilities instead of class labels for classification
        models.

        :param list[Mol] x: input to compute predictions for

        :param str | None accelerator: hardware to use for predictions, if None
            it is kept as training settings, defaults to None

        :param int | None devices: how many resources to use, if None
            it is kept as training settings, defaults to None

        :param int | None batch_size: batch size to use, if None
            it is kept as training settings, defaults to None

        :return np.ndarray: class probabilities for the input
        """
        preds = self._inner_predict(x, accelerator, devices, batch_size)
        return preds.numpy()


@ScikitLearnModelRegistry.register()
class ChempropRegressor(BaseScikitLearnModel, ScikitLearnRegressorMixin, ChempropMixin):
    """Chemprop (Directed Message Passing Neural Network) regressor.

    Uses a directed message-passing neural network (D-MPNN) that operates on
    molecular graphs for continuous property prediction. Compatible with
    regression datasets only.

    Inherits from :class:`~matcha.sklearn.base_sklearn_model.BaseScikitLearnModel`
    and :class:`ChempropMixin` for Chemprop-specific datamodule handling.

    References:

    - Yang et al., *Analyzing Learned Molecular Representations for Property Prediction*
      https://arxiv.org/abs/1904.01561

    Example usage:

    .. code-block:: python

        model = ChempropRegressor()
        model.fit(train_mols, train_y)
        predictions = model.predict(test_mols)

    :param int enc_num_layers: number of message-passing layers, defaults to 3
    :param int enc_atom_hidden_dim: hidden dimensionality in the encoder,
        defaults to 300
    :param str enc_readout: encoder readout aggregation, defaults to 'norm'
    :param float enc_dropout: dropout rate in the encoder, defaults to 0.1
    :param str enc_activation: activation function in the encoder, defaults to 'relu'
    :param int pred_hidden_dim: hidden dimensionality in the predictor FFN,
        defaults to 300
    :param int pred_num_layers: number of FFN layers in the predictor, defaults to 2
    :param str pred_activation: activation function in the predictor, defaults to 'relu'
    :param float pred_dropout: dropout rate in the predictor, defaults to 0.1
    :param int num_endpoints: number of endpoints to predict, defaults to 1
    :param str loss_fn: loss function to optimize, defaults to 'mse'
    :param str optimizer: optimizer to use, defaults to 'chemprop'
    :param dict optimizer_args: optimizer arguments, defaults to {'lr': 1e-4}
    :param str scheduler: learning rate scheduler, defaults to 'chemprop'
    :param dict scheduler_args: scheduler arguments, defaults to
        {'warmup_epochs': 2, 'max_lr': 1e-3, 'final_lr': 1e-5}
    :param int num_epochs: number of training epochs, defaults to 20
    :param int batch_size: batch size for training and prediction, defaults to 64
    :param bool stochastic_weight_averaging: whether to use SWA, defaults to False
    :param bool early_stopping: whether to use early stopping, defaults to True
    :param int patience: epochs to wait before early stopping, defaults to 10
    :param int devices: number of devices for training, defaults to 1
    :param str accelerator: hardware accelerator, defaults to 'gpu'
    :param list[str] | None feature_list: molecular feature set to compute,
        defaults to ['rdkit_all_descriptors']
    :param bool clip: whether to clip predictions to training label range,
        defaults to True
    :param dict label_encoder_params: label encoder parameters, defaults to {}
    :param str | list[str] | dict | None label_transform_map: label transform
        specification, defaults to None
    :param str scaler_type: type of target scaler, defaults to 'standard'
    :param int seed: random seed, defaults to 0
    """

    def __init__(
        self,
        enc_num_layers: int = 3,
        enc_atom_hidden_dim: int = 300,
        enc_readout: str = "norm",
        enc_dropout: float = 0.1,
        enc_activation: str = "relu",
        pred_hidden_dim: int = 300,
        pred_num_layers: int = 2,
        pred_activation: str = "relu",
        pred_dropout: float = 0.1,
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        optimizer: str = "chemprop",
        optimizer_args: dict = {"lr": 1e-4},
        scheduler: str = "chemprop",
        scheduler_args: dict = {"warmup_epochs": 2, "max_lr": 1e-3, "final_lr": 1e-5},
        num_epochs: int = 20,
        batch_size: int = 64,
        stochastic_weight_averaging: bool = False,
        early_stopping: bool = True,
        patience: int = 10,
        devices: int = 1,
        accelerator: str = "gpu",
        feature_list: list[str] | None = ["rdkit_all_descriptors"],
        clip: bool = True,
        label_encoder_params: dict = {},
        label_transform_map: str | list[str] | dict | None = None,
        scaler_type: str = "standard",
        seed: int = 0,
    ):
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        self._architecture = ChempropModel
        super(ChempropRegressor, self).__init__(params)
        silence_nuisance_warnings()

    def _adapt_dict_for_modality(self, datamodule_params, model_params):
        return ChempropMixin._adapt_dict_for_modality(
            self, datamodule_params, model_params
        )

    def _create_datamodule(self, datamodule_params, train_params):
        return ChempropMixin._create_datamodule(self, datamodule_params, train_params)
