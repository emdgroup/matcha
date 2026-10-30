"""Base class for graph-based molecular encoders with positional encoding support."""

import torch
from torch import nn
from torch_geometric.data import Batch

from matcha.torch.encoders.base_encoder import BaseEncoder
from matcha.nn.layers import LnBnDr
from matcha.nn.readouts import ReadoutRegistry


class BaseGraphEncoder(BaseEncoder):
    """Base class for all graph encoders. It is not meant to be instantiated directly,
    but rather to be used as a parent class for each featurizer.
    Subclasses from :class:`BaseEncoder` to ensure that a :method:`forward` method
    is present in children classes.
    The class is just used to automate some tedious processes related to PyTorch Geometric
    (see :method:`process_graph_batch`), handle different readout functions and jumping
    knowledge settings.

    Concrete subclasses implement :meth:`forward_nodes_per_layer` to produce
    per-layer node embeddings; the shared :meth:`forward` template method then
    combines them via the configured jumping-knowledge strategy and the readout
    to produce a graph-level embedding. This structural contract is what
    prevents architecture drift between classic and pretraining models — both
    consume the same encoder, and pretraining hooks read the per-layer output
    of the canonical encoder rather than re-implementing the layer stack.
    """

    def __init__(
        self,
        laplacian_k: int,
        rwse_k: int,
        elstatic_k: int,
        distmat_k: int,
        rrwp_k: int,
    ):
        """Initialize graph encoder with positional encoding MLPs.

        :param int laplacian_k: Dimension of Laplacian eigenvector positional encoding (0 to disable).
        :param int rwse_k: Dimension of random walk structural encoding (0 to disable).
        :param int elstatic_k: Dimension of electrostatic positional encoding (0 to disable).
        :param int distmat_k: Dimension of distance matrix positional encoding (0 to disable).
        :param int rrwp_k: Dimension of relative random walk probabilities edge encoding (0 to disable).
        """
        super().__init__()

        if laplacian_k != 0:
            self.laplacian_mlp = nn.Sequential(
                nn.BatchNorm1d(laplacian_k),
                LnBnDr(
                    laplacian_k,
                    laplacian_k * 2,
                    dropout=0.0,
                    activation="relu",
                    norm=None,
                ),
                LnBnDr(
                    laplacian_k * 2,
                    laplacian_k,
                    dropout=0.0,
                    activation=None,
                    norm=None,
                ),
                nn.Dropout(0.1),
            )
        else:
            self.laplacian_mlp = None

        if rwse_k != 0:
            self.rwse_mlp = nn.Sequential(
                nn.BatchNorm1d(rwse_k),
                LnBnDr(rwse_k, rwse_k * 2, dropout=0.0, activation="relu", norm=None),
                LnBnDr(rwse_k * 2, rwse_k, dropout=0.0, activation=None, norm=None),
                nn.Dropout(0.1),
            )
        else:
            self.rwse_mlp = None

        if elstatic_k != 0:
            self.elstatic_mlp = nn.Sequential(
                nn.BatchNorm1d(elstatic_k),
                LnBnDr(
                    elstatic_k,
                    elstatic_k * 2,
                    dropout=0.0,
                    activation="relu",
                    norm=None,
                ),
                LnBnDr(
                    elstatic_k * 2,
                    elstatic_k,
                    dropout=0.0,
                    activation=None,
                    norm=None,
                ),
                nn.Dropout(0.1),
            )
        else:
            self.elstatic_mlp = None

        if distmat_k != 0:
            self.distmat_mlp = nn.Sequential(
                nn.BatchNorm1d(distmat_k),
                LnBnDr(
                    distmat_k, distmat_k * 2, dropout=0.0, activation="relu", norm=None
                ),
                LnBnDr(
                    distmat_k * 2, distmat_k, dropout=0.0, activation=None, norm=None
                ),
                nn.Dropout(0.1),
            )
        else:
            self.distmat_mlp = None

        if rrwp_k != 0:
            self.rrwp_mlp = nn.Sequential(
                nn.BatchNorm1d(rrwp_k),
                LnBnDr(rrwp_k, rrwp_k * 2, dropout=0.0, activation="relu", norm=None),
                LnBnDr(rrwp_k * 2, rrwp_k, dropout=0.0, activation=None, norm=None),
                nn.Dropout(0.1),
            )
        else:
            self.rrwp_mlp = None

    @property
    def jk(self) -> str:
        """Jumping knowledge strategy ('last', 'concat', 'max', or 'sum').

        :returns: The configured jumping knowledge mode.
        :rtype: str
        """
        return self._jk

    def _process_graph_batch(
        self, batch: Batch
    ) -> tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and preprocess tensors from a PyG batched graph.

        Clones node and edge features, applies positional encoding MLPs
        (Laplacian, RWSE, electrostatic, distance matrix, RRWP), and
        concatenates the resulting encodings to the feature tensors.

        :param Batch batch: A PyTorch Geometric batched graph from the dataloader.
        :returns: A tuple of (batch, node_feats, edge_feats, graph_id).
        :rtype: tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # Extract node and edge features (clone to avoid modifying original)
        node_feats = batch.x.clone()
        edge_feats = batch.edge_attr.clone() if batch.edge_attr is not None else None
        graph_id = batch.batch  # Graph assignment for each node

        if hasattr(batch, "spd") and batch.spd is not None:
            batch.spd = batch.spd.to(node_feats.device)

        node_pe = []
        edge_pe = []

        if self.laplacian_mlp is not None:
            laplacian = self.laplacian_mlp(batch.laplacian_k)
            node_pe.append(laplacian)

        if self.rwse_mlp is not None:
            rwse = self.rwse_mlp(batch.rwse_k)
            node_pe.append(rwse)

        if self.elstatic_mlp is not None:
            elstatic = self.elstatic_mlp(batch.elstatic_k)
            node_pe.append(elstatic)

        if self.distmat_mlp is not None:
            distmat = self.distmat_mlp(batch.distmat_k)
            node_pe.append(distmat)

        if len(node_pe) > 0:
            node_pe = torch.concat(node_pe, axis=1)
            node_feats = torch.concat([node_feats, node_pe], axis=1)

        if self.rrwp_mlp is not None:
            rrwp = self.rrwp_mlp(batch.rrwp_k)
            edge_pe.append(rrwp)

        if len(edge_pe) > 0:
            edge_pe = torch.concat(edge_pe, axis=1)
            edge_feats = torch.concat([edge_feats, edge_pe], axis=1)

        return batch, node_feats, edge_feats, graph_id

    def _parse_readout(self, readout: str, **kwargs):
        """Utility function to handle readout function instantiation.

        Readouts that require input dimensions will use self.fp_dim.
        Some readouts (like set2set) modify the output dimension.

        :param str readout: Name of the readout function.
        :param kwargs: Additional keyword arguments passed to the readout.
        """
        # Readouts that double output dimension
        if readout == "set2set":
            self.readout = ReadoutRegistry[readout](
                in_channels=self.fp_dim,
                processing_steps=kwargs.get("processing_steps", 3),
                num_layers=kwargs.get("num_layers", 1),
            )
            self._fp_dim = self.fp_dim * 2
        # Readouts that require input dimension
        elif readout == "attentive":
            self.readout = ReadoutRegistry[readout](gate_nn_channels=self.fp_dim)
        elif readout in ("lstm", "gru"):
            out_channels = kwargs.get("out_channels", self.fp_dim)
            self.readout = ReadoutRegistry[readout](
                in_channels=self.fp_dim,
                out_channels=out_channels,
            )
            self._fp_dim = out_channels
        elif readout == "sort":
            k = kwargs.get("k", 10)
            self.readout = ReadoutRegistry[readout](k=k)
            self._fp_dim = self.fp_dim * k
        elif readout == "softmax":
            self.readout = ReadoutRegistry[readout](
                learn=kwargs.get("learn", True),
                t=kwargs.get("t", 1.0),
            )
        elif readout == "powermean":
            self.readout = ReadoutRegistry[readout](
                learn=kwargs.get("learn", True),
                p=kwargs.get("p", 1.0),
            )
        elif readout == "quantile":
            self.readout = ReadoutRegistry[readout](q=kwargs.get("q", 0.5))
        elif readout == "multi":
            aggrs = ["mean", "max", "sum", "var"]
            self.readout = ReadoutRegistry[readout](aggrs=aggrs, mode="attn")
        # Simple readouts that don't require parameters
        else:
            self.readout = ReadoutRegistry[readout]()

    def _parse_jk(self, jk: str):
        """Configure jumping knowledge strategy and update :attr:`fp_dim` accordingly.

        :param str jk: Strategy name — 'concat' multiplies ``fp_dim`` by ``num_layers``;
            'last', 'max', and 'sum' keep ``fp_dim`` equal to ``atom_hidden_dim``.
        """
        self._jk = jk
        if self.jk == "concat":
            self._fp_dim = self.hparams["atom_hidden_dim"] * self.hparams["num_layers"]
        else:
            self._fp_dim = self.hparams["atom_hidden_dim"]

    def _run_jk(self, atom_features: torch.Tensor) -> torch.Tensor:
        """Combine per-layer atom representations using the configured jumping knowledge strategy.

        :param list[torch.Tensor] atom_features: List of node feature tensors, one per
            message-passing layer, each of shape ``[num_nodes, atom_hidden_dim]``.
        :returns: Combined node features with shape determined by the JK strategy.
        :rtype: torch.Tensor
        """
        if self.jk == "concat":
            final_atom_features = torch.cat(atom_features, dim=1)
        elif self.jk == "last":
            final_atom_features = atom_features[-1]
        elif self.jk == "max":
            atom_features = [h.unsqueeze(0) for h in atom_features]
            final_atom_features = torch.max(torch.cat(atom_features, dim=0), dim=0)[0]
        elif self.jk == "sum":
            atom_features = [h.unsqueeze(0) for h in atom_features]
            final_atom_features = torch.sum(torch.cat(atom_features, dim=0), dim=0)
        return final_atom_features

    def forward_nodes_per_layer(self, graph: Batch) -> tuple[list[torch.Tensor], Batch]:
        """Run message passing and return one node-feature tensor per layer.

        Concrete graph encoders should implement this to expose the
        intermediate node representations produced by every message-passing
        layer. It is the single hook consumed by both the classic path (via
        :meth:`forward`) and the pretraining path (via
        :class:`BaseGraphPretrainingModel`), so the same layer stack is
        guaranteed to drive both.

        The default implementation raises :class:`NotImplementedError` so
        that encoders which have not yet migrated to the shared contract
        (rolled out per-architecture across the issue #24 stages) still work
        on the classic path but fail loudly if the pretraining path is
        wired against them. Once every :class:`BaseGraphEncoder` subclass
        implements this method, the base can be tightened to
        ``@abstractmethod``.

        :param Batch graph: Batched PyG graph from the dataloader.
        :returns: A tuple of ``(all_atom_feats, g)`` where ``all_atom_feats``
            is a list of length ``num_layers`` containing node features of
            shape ``[num_nodes, atom_hidden_dim]``, and ``g`` is the
            (possibly modified) PyG :class:`Batch` used for the readout.
        :rtype: tuple[list[torch.Tensor], Batch]
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented forward_nodes_per_layer; "
            "canonical + pretraining encoder unification (issue #24) is still "
            "in progress for this architecture."
        )

    def forward(self, graph: Batch) -> torch.Tensor:
        """Convert a batched PyG graph into a ``[batch_size, fp_dim]`` embedding.

        Template method that delegates the layer loop to
        :meth:`forward_nodes_per_layer`, combines the per-layer node features
        via the configured jumping-knowledge strategy, and applies the
        readout. Subclasses that have migrated to the shared contract should
        not override this method — override :meth:`forward_nodes_per_layer`
        instead. Legacy subclasses may still override :meth:`forward`
        directly until they migrate.

        :param Batch graph: Batched PyG graph from the dataloader.
        :returns: Graph-level embedding of shape ``[batch_size, fp_dim]``.
        :rtype: torch.Tensor
        """
        all_atom_feats, g = self.forward_nodes_per_layer(graph)
        final_atom_feats = self._run_jk(all_atom_feats)
        return self.readout(g, final_atom_feats)
