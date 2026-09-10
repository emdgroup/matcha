"""Chemprop (Directed Message Passing Neural Network) classic model."""

from chemprop.models.model import MPNN
from chemprop.nn import BondMessagePassing
from chemprop.nn.predictors import (
    RegressionFFN,
    BinaryClassificationFFN,
    MveFFN,
)
from chemprop.nn.agg import AggregationRegistry as ChempropAggRegistry
from chemprop.nn.agg import AttentiveAggregation
from chemprop.nn.metrics import LossFunctionRegistry as ChempropLossRegistry
from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from matcha.utils.schemas import ChempropInputModel, UncertaintyMethod
from matcha.torch.models.classic.base_classic_model import ClassicModelRegistry

# Clamp value for chemprop's softplus-variance before taking the log, so
# numerically-zero variances do not overflow into ``-inf`` log-variance.
_MVE_VAR_EPS = 1e-6


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
        uncertainty: UncertaintyMethod = "mc-dropout",
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
            uncertainty=uncertainty,
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

        if uncertainty == "mve":
            MLP = MveFFN
        elif loss_fn in ("bce", "ce"):
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

    @property
    def uncertainty_method(self) -> str:
        """Configured uncertainty method (``"mc-dropout"`` or ``"mve"``).

        Mirrors :attr:`BaseClassicModel.uncertainty_method` so
        :class:`~matcha.sklearn.managers.uncertainty_manager.UncertaintyManager`
        can dispatch on the same value regardless of the model family.
        """
        return self.hparams.get("uncertainty") or "mc-dropout"

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Prediction step returning point predictions.

        For MVE-configured models, chemprop's :class:`MveFFN` emits
        ``(N, num_endpoints, 2)`` where ``[..., 0]`` is the mean and
        ``[..., 1]`` is the softplus-variance. We slice out the mean so
        the sklearn surface sees the same ``(N, T)`` layout as any other
        regressor. Chemprop does not support MC-dropout inference, so no
        branching on ``self.mc_dropout_flag`` is needed here.
        """
        y_pred = super().predict_step(batch, batch_idx, dataloader_idx)
        if self.uncertainty_method == "mve":
            y_pred = y_pred[..., 0]
        return y_pred

    def predict_variance_step(self, batch):
        """Return the intrinsic mean and log-variance for an MVE-configured model.

        Chemprop's :class:`MveFFN` produces ``(N, num_endpoints, 2)`` with
        ``[..., 0]`` a mean and ``[..., 1]`` a softplus-parameterised
        variance (strictly positive). Matcha's
        :meth:`~matcha.sklearn.managers.uncertainty_manager.UncertaintyManager._compute_mve`
        expects a log-variance so it can exponentiate it back to a variance
        and unscale it. The softplus → log conversion is the sole adapter
        between chemprop's convention and matcha's; it is clamped by
        :data:`_MVE_VAR_EPS` to keep ``log`` finite when the softplus output
        rounds to zero.

        :param batch: chemprop batch object.
        :returns: tuple ``(mean, log_var)``, each with shape
            ``(batch, num_endpoints)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises RuntimeError: if the model was not configured with
            ``uncertainty="mve"``.
        """
        if self.uncertainty_method != "mve":
            raise RuntimeError(
                "predict_variance_step is only available when the model was "
                "instantiated with uncertainty='mve'."
            )
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        bmg, V_d, X_d, *_ = batch
        with torch.no_grad():
            y_pred = self(bmg, V_d, X_d)
        mean = y_pred[..., 0]
        log_var = torch.log(y_pred[..., 1].clamp_min(_MVE_VAR_EPS))
        return mean, log_var

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
