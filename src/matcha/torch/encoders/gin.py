"""Graph Isomorphism Network (GIN) encoder for molecular graphs."""

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import nn

from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from matcha.nn.layers import LnBnDr


@EncoderRegistry.register()
class GIN(BaseGraphEncoder, HyperparametersMixin):
    """Graph Isomorphism Network (GIN) encoder as described in 'How powerful are
    graph neural networks?'. This class performes message passing via graph convolutions,
    updates the atom representations and returns a molecular embedding which can
    then be used for molecular property prediction.
    It inherits from :class:`BaseGraphEncoder` for common graph encoding routines
    (e.g. jk-related routines) and from :class:`lightning.pytorch.core.mixins`
    for saving its hyperparameters.
    References:
    - https://arxiv.org/abs/1810.00826
    - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    - https://arxiv.org/abs/1905.12265

    It is intended to be used inside a :class:`BaseClassicModel` instance.
    Check the docs of :class:`matcha.torch.models.classic.GINModel` for further details.

    :param int num_layers: number of message passing layers

    :param int atom_input_dim: number of input atom features from GraphFeaturizer

    :param int bond_input_dim: number of input bond features from GraphFeaturizer

    :param int atom_hidden_dim: number of hidden atom features in message passing layers

    :param str activation: activation function to be used in all layers

    :param str aggregation: function to use to aggregate atom messages in a given
        neighborhood

    :param float dropout: dropout noise level

    :param str | None norm: which norm to use inside GIN layers ('batch', 'layer',
        'graph', or None), defaults to 'graph'

    :param str jk: jumping knowledge strategy to use when returning molecular
        representations after forward pass

    :param str readout: readout function to aggregate all atom representations

    :param float eps: initial value of the ``eps`` term in ``GINEConv``

    :param bool train_eps: whether to learn ``eps`` as a parameter (paper-recommended
        GIN-ε variant)
    """

    def __init__(
        self,
        num_layers: int,
        atom_input_dim: int,
        bond_input_dim: int,
        atom_hidden_dim: int,
        activation: str,
        aggregation: str,
        dropout: float,
        norm: str | None,
        jk: str,
        readout: str,
        laplacian_k: int,
        rwse_k: int,
        elstatic_k: int,
        distmat_k: int,
        rrwp_k: int,
        eps: float,
        train_eps: bool,
    ):
        super().__init__(laplacian_k, rwse_k, elstatic_k, distmat_k, rrwp_k)
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        start_dim = atom_input_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                LnBnDr(
                    start_dim,
                    atom_hidden_dim * 2,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                ),
                LnBnDr(
                    atom_hidden_dim * 2,
                    atom_hidden_dim,
                    dropout=dropout,
                    activation=None,
                    norm=norm,
                ),
            )
            self.layers.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=bond_input_dim,
                    aggr=aggregation,
                    eps=eps,
                    train_eps=train_eps,
                )
            )
            start_dim = atom_hidden_dim
        self._parse_jk(jk)
        self._parse_readout(readout)

    def forward(self, graph: Batch) -> torch.Tensor:
        """Converts a batched PyG graph into a (x, self.fp_dim) tensor for further
        processing.

        :param Batch graph: batched PyG graph from the dataloader

        :return torch.Tensor: learned representation
        """
        g, feats, efeats, _ = self._process_graph_batch(graph)
        all_atom_feats = []

        for layer in self.layers:
            feats = layer(feats, g.edge_index, efeats)
            if all_atom_feats != []:
                feats = feats + all_atom_feats[-1]
            all_atom_feats.append(feats)

        final_atom_feats = self._run_jk(all_atom_feats)

        return self.readout(g, final_atom_feats)
