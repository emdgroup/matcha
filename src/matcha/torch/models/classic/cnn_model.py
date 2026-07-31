"""1-D Convolutional Neural Network (CNN) classic model for chemical language."""

from matcha.torch.models.classic.base_classic_model import (
    BaseClassicModel,
    ClassicModelRegistry,
)
from matcha.torch.encoders.cnn import CNN
from matcha.utils.schemas import CNNInputModel

from lightning.pytorch.core.mixins import HyperparametersMixin


@ClassicModelRegistry.register()
class CNNModel(BaseClassicModel, HyperparametersMixin):
    """1-D Convolutional Neural Network (CNN) for molecular property prediction
    from chemical language (e.g. SMILES via augmented input strings).

    Uses parallel 1-D convolutions with varying kernel sizes and self-attention
    pooling. Inherits from :class:`BaseClassicModel` for common
    training/prediction routines and from
    :class:`~lightning.pytorch.core.mixins.HyperparametersMixin` for saving
    hyperparameters.

    References:

    - https://arxiv.org/abs/2407.12152
    - https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g

    Example usage:

    .. code-block:: python

        model = CNNModel(enc_num_characters=n_tokens)
        trainer = L.Trainer(max_epochs=50)
        trainer.fit(model=model, train_dataloaders=train_dataloader)

    :param int additional_mol_features_dim: dimensionality of extra molecular
        features concatenated to encoder output, defaults to 0
    :param int enc_num_characters: vocabulary size (unique tokens), defaults to 400
    :param int enc_hidden_dim: embedding and convolution hidden dimensionality,
        defaults to 256
    :param list[int] enc_kernel_dims: kernel sizes for each parallel convolution,
        defaults to [5, 9, 13, 17, 21]
    :param int enc_num_heads: number of self-attention heads for pooling, defaults to 4
    :param str enc_activation: activation function in the encoder, defaults to 'swish'
    :param float enc_dropout: dropout rate in the encoder, defaults to 0.2
    :param list[int] pred_hidden_dims: hidden layer sizes in the MLP predictor,
        defaults to [512, 256]
    :param list[int] | None pred_task_head_dims: per-task head dimensions, defaults to None
    :param str pred_activation: activation in the predictor, defaults to 'swish'
    :param float pred_dropout: dropout rate in the predictor, defaults to 0.2
    :param int num_endpoints: number of prediction targets, defaults to 1
    :param str loss_fn: loss function name, defaults to 'mse'
    :param dict loss_args: additional loss function arguments, defaults to {}
    :param str optimizer: optimizer name, defaults to 'adam'
    :param dict optimizer_args: additional optimizer arguments, defaults to {'lr': 1e-3}
    :param str scheduler: learning rate scheduler name, defaults to 'cosine_annealing'
    :param dict scheduler_args: additional scheduler arguments,
        defaults to {'min_lr': 1e-6, 'total_steps': 50}
    """

    def __init__(
        self,
        additional_mol_features_dim: int = 0,
        enc_num_characters: int = 400,
        enc_hidden_dim: int = 256,
        enc_kernel_dims: list[int] = [5, 9, 13, 17, 21],
        enc_num_heads: int = 4,
        enc_activation: str = "swish",
        enc_dropout: float = 0.2,
        pred_hidden_dims: list[int] = [512, 256],
        pred_task_head_dims: list[int] | None = None,
        pred_activation: str = "swish",
        pred_dropout: float = 0.2,
        num_endpoints: int = 1,
        loss_fn: str = "mse",
        loss_args: dict = {},
        optimizer: str = "adam",
        optimizer_args: dict = {"lr": 1e-3},
        scheduler: str = "cosine_annealing",
        scheduler_args: dict = {"min_lr": 1e-6, "total_steps": 50},
    ):
        super().__init__(additional_mol_features_dim=additional_mol_features_dim)
        self.save_hyperparameters()
        self.params = CNNInputModel(**self.hparams)

        self.encoder = CNN(
            num_characters=enc_num_characters,
            embedding_dim=enc_hidden_dim,
            hidden_dim=enc_hidden_dim,
            kernel_dims=enc_kernel_dims,
            num_heads=enc_num_heads,
            activation=enc_activation,
            dropout=enc_dropout,
        )
        self._parse_predictor()
        self._parse_train_config()
