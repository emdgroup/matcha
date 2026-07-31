"""Chemprop (Directed Message Passing Neural Network) classic model."""

from chemprop.models.model import MPNN
from chemprop.nn import BondMessagePassing
from chemprop.nn.predictors import (
    RegressionFFN,
    BinaryClassificationFFN,
)
from chemprop.nn.agg import AggregationRegistry as ChempropAggRegistry
from chemprop.nn.agg import AttentiveAggregation
from chemprop.nn.metrics import LossFunctionRegistry as ChempropLossRegistry
from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from matcha.utils.schemas import ChempropInputModel
from matcha.torch.models.classic.base_classic_model import ClassicModelRegistry


@ClassicModelRegistry.register()
class ChempropModel(MPNN, HyperparametersMixin):
    """Chemprop Directed Message Passing Neural Network (D-MPNN) for molecular
    property prediction.

    Wraps the Chemprop MPNN implementation with matcha's registry and
    hyperparameter management. Unlike other classic models, this class inherits
    directly from :class:`chemprop.models.model.MPNN` and uses Chemprop's
    built-in scheduler and optimizer by default.

    Reference: Yang et al., *Analyzing Learned Molecular Representations for
    Property Prediction* (https://arxiv.org/abs/1904.01561)

    Example usage:

    .. code-block:: python

        model = ChempropModel(num_endpoints=1)
        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param int enc_atom_hidden_dim: hidden dimensionality of bond message passing,
        defaults to 300
    :param int enc_num_layers: depth of message passing, defaults to 3
    :param float enc_dropout: dropout in message-passing layers, defaults to 0.2
    :param str enc_activation: activation function in the encoder, defaults to 'relu'
    :param str enc_readout: aggregation strategy ('norm', 'mean', 'sum', 'attentive'),
        defaults to 'norm'
    :param int additional_mol_features_dim: dimensionality of extra molecular
        features concatenated to the learned fingerprint, defaults to 0
    :param int pred_hidden_dim: hidden size of feed-forward predictor layers,
        defaults to 300
    :param int pred_num_layers: number of feed-forward predictor layers, defaults to 2
    :param float pred_dropout: dropout in predictor layers, defaults to 0.2
    :param str pred_activation: activation in predictor layers, defaults to 'relu'
    :param int num_endpoints: number of prediction targets, defaults to 1
    :param str loss_fn: loss function name ('mse', 'bce', 'ce'),
        defaults to 'mse'
    :param str optimizer: optimizer name, defaults to 'chemprop'
    :param dict optimizer_args: optimizer arguments, defaults to {'lr': 1e-3}
    :param str scheduler: scheduler name, defaults to 'chemprop'
    :param dict scheduler_args: scheduler arguments,
        defaults to {'warmup_epochs': 5, 'max_lr': 1e-2, 'final_lr': 1e-5}
    """

    def __init__(
        self,
        enc_atom_hidden_dim: int = 300,
        enc_num_layers: int = 3,
        enc_dropout: float = 0.2,
        enc_activation: str = "relu",
        enc_readout: str = "norm",
        additional_mol_features_dim: int = 0,
        pred_hidden_dim: int = 300,
        pred_num_layers: int = 2,
        pred_dropout: float = 0.2,
        pred_activation: str = "relu",
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        optimizer: str = "chemprop",
        optimizer_args: dict = {"lr": 1e-3},
        scheduler: str = "chemprop",
        scheduler_args: dict = {"warmup_epochs": 5, "max_lr": 1e-2, "final_lr": 1e-5},
    ):
        self.save_hyperparameters()
        self.params = ChempropInputModel(
            enc_atom_hidden_dim=enc_atom_hidden_dim,
            enc_num_layers=enc_num_layers,
            enc_dropout=enc_dropout,
            enc_activation=enc_activation,
            enc_readout=enc_readout,
            additional_mol_features_dim=additional_mol_features_dim,
            pred_hidden_dim=pred_hidden_dim,
            pred_num_layers=pred_num_layers,
            pred_dropout=pred_dropout,
            pred_activation=pred_activation,
            num_endpoints=num_endpoints,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
        )
        mp = BondMessagePassing(
            d_h=enc_atom_hidden_dim,
            depth=enc_num_layers,
            dropout=enc_dropout,
            activation=enc_activation,
        )

        if enc_readout in ChempropAggRegistry:
            agg = ChempropAggRegistry[enc_readout]()
        elif enc_readout == "attentive":
            agg = AttentiveAggregation(output_size=enc_atom_hidden_dim)

        if loss_fn in ("bce", "ce"):
            MLP = BinaryClassificationFFN
        else:
            MLP = RegressionFFN

        mlp = MLP(
            input_dim=enc_atom_hidden_dim + additional_mol_features_dim,
            hidden_dim=pred_hidden_dim,
            n_layers=pred_num_layers,
            dropout=pred_dropout,
            activation=pred_activation,
            n_tasks=num_endpoints,
            criterion=ChempropLossRegistry[loss_fn](),
        )

        super().__init__(
            mp,
            agg,
            mlp,
            batch_norm=True,
            warmup_epochs=scheduler_args["warmup_epochs"],
            init_lr=optimizer_args["lr"],
            max_lr=scheduler_args["max_lr"],
            final_lr=scheduler_args["final_lr"],
        )

    def compute_learned_embedding(self, dataloader) -> list:
        """Extract learned fingerprints for all batches in a dataloader.

        :param dataloader: dataloader yielding Chemprop batch objects
        :returns: list of fingerprint tensors, one per batch
        :rtype: list[torch.Tensor]
        """
        with torch.no_grad():
            fingerprints = [
                self.encoding(batch.bmg, batch.V_d, batch.X_d, i=-1)
                for batch in dataloader
            ]
        return fingerprints

    def encode(self, batch):
        """Encode a single batch into learned molecular fingerprints.

        :param batch: Chemprop batch object
        :returns: molecular fingerprint tensor
        :rtype: torch.Tensor
        """
        return self.encoding(batch, batch.V_d, batch.X_d, i=-1)

    def switch_mc_dropout(self, *args, **kwargs):
        """Not supported for Chemprop models.

        :raises ValueError: always, as Chemprop does not support MC Dropout
        """
        raise ValueError("Chemprop does not support MC Dropout")

    def set_label_names(self, label_names: list[str]):
        """Set endpoint label names for logging and metric tracking.

        :param list[str] label_names: names of the prediction endpoints
        """
        self._label_names = label_names
