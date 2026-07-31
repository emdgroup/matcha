"""Base encoder interface for all molecular representation encoders."""

import torch
from abc import abstractmethod, ABC
from matcha.utils.registry import ClassRegistry
from lightning import LightningModule


class BaseEncoder(LightningModule, ABC):
    """Abstract base class for all encoders.

    Defines the interface that all encoder subclasses must implement. Enforces
    the presence of a :meth:`forward` method that maps featurized molecular
    inputs to fixed-size embedding tensors.

    Not meant to be instantiated directly; use a concrete subclass such as
    :class:`GIN`, :class:`GPS`, :class:`CNN`, etc.
    """

    def __init__(self):
        super().__init__()

    @property
    def fp_dim(self) -> int:
        """Output fingerprint (embedding) dimensionality.

        :returns: The size of the last dimension of the tensor returned by :meth:`forward`.
        :rtype: int
        """
        return self._fp_dim

    @abstractmethod
    def forward(self, x: object) -> torch.Tensor:
        """This method needs to convert the batched output from one of the featurizers
        into a torch.Tensor of shape (batch_size, K), which can then be processed by
        a MLP-like architecture for prediction.

        :param object x: The batched output from any featurizer. Depending on the featurizer,
            it can be different things, e.g. torch.Tensor,  list, ...

        :return torch.Tensor: a tensor of shape (batch_size, K), corresponding to the
            learned embeddings for the batch_size molecules.
        """


EncoderRegistry = ClassRegistry[BaseEncoder]()
