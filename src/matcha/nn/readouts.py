"""Graph readout (pooling) layers registered in the :data:`ReadoutRegistry`."""

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import aggr

from matcha.utils.registry import ClassRegistry

ReadoutRegistry = ClassRegistry()


class PyGAggregationWrapper(nn.Module):
    """Wrapper to adapt PyG aggregation classes to the matcha readout interface.

    PyG aggregations expect (x, index) while matcha readouts expect (graph, x).
    This wrapper handles the conversion.
    """

    def __init__(self, aggregation: aggr.Aggregation):
        """
        :param aggregation: A PyG aggregation instance to wrap.
        :type aggregation: torch_geometric.nn.aggr.Aggregation
        """
        super().__init__()
        self.aggregation = aggregation

    def forward(self, graph: Batch, x: torch.Tensor) -> torch.Tensor:
        """
        :param graph: PyG batch containing the ``batch`` assignment vector.
        :type graph: torch_geometric.data.Batch
        :param torch.Tensor x: Node features of shape ``(total_nodes, feat_dim)``.
        :returns: Graph-level features of shape ``(num_graphs, feat_dim)``.
        :rtype: torch.Tensor
        """
        return self.aggregation(x, graph.batch)


# Simple aggregations
@ReadoutRegistry.register("sum")
class SumPooling(PyGAggregationWrapper):
    """Sum aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.SumAggregation())


@ReadoutRegistry.register("mean")
class MeanPooling(PyGAggregationWrapper):
    """Mean aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.MeanAggregation())


@ReadoutRegistry.register("max")
class MaxPooling(PyGAggregationWrapper):
    """Max aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.MaxAggregation())


@ReadoutRegistry.register("min")
class MinPooling(PyGAggregationWrapper):
    """Min aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.MinAggregation())


@ReadoutRegistry.register("mul")
class MulPooling(PyGAggregationWrapper):
    """Multiplicative aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.MulAggregation())


# Statistical aggregations
@ReadoutRegistry.register("var")
class VarPooling(PyGAggregationWrapper):
    """Variance aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.VarAggregation())


@ReadoutRegistry.register("std")
class StdPooling(PyGAggregationWrapper):
    """Standard deviation aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.StdAggregation())


@ReadoutRegistry.register("median")
class MedianPooling(PyGAggregationWrapper):
    """Median aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.MedianAggregation())


@ReadoutRegistry.register("vpa")
class VariancePreservingPooling(PyGAggregationWrapper):
    """Variance-preserving aggregation over node features."""

    def __init__(self):
        super().__init__(aggr.VariancePreservingAggregation())


@ReadoutRegistry.register("quantile")
class QuantilePooling(PyGAggregationWrapper):
    """Quantile aggregation over node features.

    :param float q: The quantile to compute (between 0 and 1).
    """

    def __init__(self, q: float = 0.5):
        super().__init__(aggr.QuantileAggregation(q=q))


# Learnable aggregations
@ReadoutRegistry.register("softmax")
class SoftmaxPooling(PyGAggregationWrapper):
    """Softmax aggregation based on a temperature term.

    :param bool learn: Whether to learn the temperature parameter.
    :param float t: Initial temperature value.
    :param bool semi_grad: If True, does not compute gradients for temperature.
    """

    def __init__(self, learn: bool = True, t: float = 1.0, semi_grad: bool = False):
        super().__init__(aggr.SoftmaxAggregation(learn=learn, t=t, semi_grad=semi_grad))


@ReadoutRegistry.register("powermean")
class PowerMeanPooling(PyGAggregationWrapper):
    """Power mean aggregation based on a power term.

    :param bool learn: Whether to learn the power parameter.
    :param float p: Initial power value.
    """

    def __init__(self, learn: bool = True, p: float = 1.0):
        super().__init__(aggr.PowerMeanAggregation(learn=learn, p=p))


# Sequence-based aggregations
@ReadoutRegistry.register("lstm")
class LSTMPooling(PyGAggregationWrapper):
    """LSTM-style aggregation treating nodes as a sequence.

    :param int in_channels: Size of each input sample.
    :param int out_channels: Size of each output sample.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr.LSTMAggregation(in_channels, out_channels))


@ReadoutRegistry.register("gru")
class GRUPooling(PyGAggregationWrapper):
    """GRU aggregation treating nodes as a sequence.

    :param int in_channels: Size of each input sample.
    :param int out_channels: Size of each output sample.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr.GRUAggregation(in_channels, out_channels))


# Set-based aggregations
@ReadoutRegistry.register("set2set")
class Set2SetPooling(PyGAggregationWrapper):
    """Set2Set aggregation based on iterative content-based attention.

    :param int in_channels: Size of each input sample.
    :param int processing_steps: Number of processing steps.
    :param int num_layers: Number of recurrent layers.
    """

    def __init__(
        self, in_channels: int, processing_steps: int = 3, num_layers: int = 1
    ):
        super().__init__(
            aggr.Set2Set(
                in_channels, processing_steps=processing_steps, num_layers=num_layers
            )
        )


@ReadoutRegistry.register("sort")
class SortPooling(PyGAggregationWrapper):
    """Sort aggregation where node features are sorted in descending order.

    :param int k: The number of nodes to hold for each graph.
    """

    def __init__(self, k: int):
        super().__init__(aggr.SortAggregation(k=k))


# Attention-based aggregations
@ReadoutRegistry.register("attentive")
class AttentivePooling(PyGAggregationWrapper):
    """Soft attention aggregation from Graph Matching Networks.

    :param int gate_nn_channels: Hidden channels for the gate network.
    :param int nn_channels: Hidden channels for the feature network (optional).
    """

    def __init__(self, gate_nn_channels: int, nn_channels: int | None = None):
        gate_nn = nn.Sequential(
            nn.Linear(gate_nn_channels, gate_nn_channels),
            nn.ReLU(),
            nn.Linear(gate_nn_channels, 1),
        )
        if nn_channels is not None:
            feat_nn = nn.Sequential(
                nn.Linear(nn_channels, nn_channels),
                nn.ReLU(),
                nn.Linear(nn_channels, nn_channels),
            )
        else:
            feat_nn = None
        super().__init__(aggr.AttentionalAggregation(gate_nn=gate_nn, nn=feat_nn))


@ReadoutRegistry.register("graphmultiset")
class GraphMultisetTransformerPooling(PyGAggregationWrapper):
    """Graph Multiset Transformer pooling.

    :param int in_channels: Size of each input sample.
    :param int hidden_channels: Size of hidden layer.
    :param int out_channels: Size of each output sample.
    :param int num_heads: Number of attention heads.
    :param int pool_sequences: Pool sequences parameter.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        pool_sequences: str | list[str] = ["GMPool_G", "SelfAtt", "GMPool_I"],
    ):
        super().__init__(
            aggr.GraphMultisetTransformer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                pool_sequences=pool_sequences,
            )
        )


# MLP-based aggregations
@ReadoutRegistry.register("mlp")
class MLPPooling(PyGAggregationWrapper):
    """MLP aggregation where elements are flattened and processed by an MLP.

    :param int in_channels: Size of each input sample.
    :param int out_channels: Size of each output sample.
    :param int max_num_elements: Maximum number of elements to aggregate.
    :param int hidden_channels: Hidden layer size (optional).
    :param int num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_num_elements: int,
        hidden_channels: int | None = None,
        num_layers: int = 1,
    ):
        super().__init__(
            aggr.MLPAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                max_num_elements=max_num_elements,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
        )


@ReadoutRegistry.register("deepsets")
class DeepSetsPooling(PyGAggregationWrapper):
    """Deep Sets aggregation with element-wise MLP, sum, and final MLP.

    :param int in_channels: Size of each input sample.
    :param int out_channels: Size of each output sample.
    :param int hidden_channels: Hidden layer size (optional).
    :param int num_layers: Number of MLP layers in each network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        num_layers: int = 1,
    ):
        super().__init__(
            aggr.DeepSetsAggregation(
                local_nn=nn.Sequential(
                    nn.Linear(in_channels, hidden_channels or out_channels),
                    nn.ReLU(),
                    *(
                        [
                            nn.Linear(
                                hidden_channels or out_channels,
                                hidden_channels or out_channels,
                            ),
                            nn.ReLU(),
                        ]
                        * (num_layers - 1)
                    ),
                ),
                global_nn=nn.Sequential(
                    nn.Linear(hidden_channels or out_channels, out_channels),
                ),
            )
        )


@ReadoutRegistry.register("settransformer")
class SetTransformerPooling(PyGAggregationWrapper):
    """Set Transformer aggregation with multi-head attention blocks.

    :param int in_channels: Size of each input sample.
    :param int out_channels: Size of each output sample.
    :param int hidden_channels: Size of hidden layer (optional).
    :param int num_heads: Number of attention heads.
    :param int num_seed_points: Number of seed vectors.
    :param int num_encoder_blocks: Number of encoder blocks.
    :param int num_decoder_blocks: Number of decoder blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        num_heads: int = 4,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 1,
        num_decoder_blocks: int = 1,
    ):
        super().__init__(
            aggr.SetTransformerAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                num_seed_points=num_seed_points,
                num_encoder_blocks=num_encoder_blocks,
                num_decoder_blocks=num_decoder_blocks,
            )
        )


@ReadoutRegistry.register("lcm")
class LCMPooling(PyGAggregationWrapper):
    """Learnable Commutative Monoid aggregation using binary tree reduction.

    :param int in_channels: Size of each input sample.
    :param int out_channels: Size of each output sample.
    :param int project: Whether to project inputs/outputs.
    """

    def __init__(self, in_channels: int, out_channels: int, project: bool = True):
        super().__init__(
            aggr.LCMAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                project=project,
            )
        )


# Multi-aggregation
@ReadoutRegistry.register("multi")
class MultiPooling(PyGAggregationWrapper):
    """Combines multiple aggregations.

    :param list[str] aggrs: List of aggregation names to combine.
    :param str mode: Combination mode ('cat', 'proj', 'attn', 'sum', 'mean', 'max', 'min', 'logsumexp', 'std', 'var').
    :param dict mode_kwargs: Additional kwargs for the combination mode.
    """

    def __init__(
        self,
        aggrs: list[str],
        mode: str = "attn",
        mode_kwargs: dict | None = None,
    ):
        super().__init__(
            aggr.MultiAggregation(
                aggrs=aggrs,
                mode=mode,
                mode_kwargs=mode_kwargs or {},
            )
        )


# Degree-scaled aggregation
@ReadoutRegistry.register("degreescaler")
class DegreeScalerPooling(PyGAggregationWrapper):
    """Combines aggregators with degree-based scalers (PNA-style).

    :param list[str] aggrs: List of aggregation names to combine.
    :param list[str] scalers: List of scaler names ('identity', 'amplification', 'attenuation', 'linear', 'inverse_linear').
    :param float deg: Degree tensor or average degree.
    """

    def __init__(
        self,
        aggrs: list[str],
        scalers: list[str],
        deg: torch.Tensor,
    ):
        super().__init__(
            aggr.DegreeScalerAggregation(
                aggr=aggrs,
                scaler=scalers,
                deg=deg,
            )
        )


# Virtual node pooling (kept from original)
@ReadoutRegistry.register("virtualnode")
class VirtualNodePooling(nn.Module):
    """Extracts the virtual node representation as the graph embedding.

    Assumes virtual nodes are added at the end of each graph.
    """

    def __init__(self):
        super(VirtualNodePooling, self).__init__()

    def forward(self, graph: Batch, feat: torch.Tensor) -> torch.Tensor:
        """
        :param graph: PyG batch with a ``ptr`` attribute.
        :type graph: torch_geometric.data.Batch
        :param torch.Tensor feat: Node features of shape ``(total_nodes, feat_dim)``.
        :returns: Virtual-node features of shape ``(num_graphs, feat_dim)``.
        :rtype: torch.Tensor
        """
        # In PyG, ptr gives indices where each graph starts
        # Virtual nodes are at the end of each graph (ptr[1:] - 1)
        virtual_node_indices = graph.ptr[1:] - 1
        virtual_node_feats = feat[virtual_node_indices]
        return virtual_node_feats
