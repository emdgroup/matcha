"""Tests for matcha.nn.readouts – ReadoutRegistry and PyGAggregationWrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock, sentinel

import pytest
import torch


# ===================================================================
# Imports (skip entire module if torch_geometric unavailable)
# ===================================================================

pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.nn import aggr  # noqa: E402

from matcha.nn import readouts as readout_module  # noqa: E402
from matcha.nn.readouts import (  # noqa: E402
    PyGAggregationWrapper,
    ReadoutRegistry,
    VirtualNodePooling,
)


# ===================================================================
# Registry completeness
# ===================================================================


class TestReadoutRegistry:
    def test_required_aliases_resolve_to_expected_classes(self):
        expected = {
            "sum": "SumPooling",
            "mean": "MeanPooling",
            "max": "MaxPooling",
            "min": "MinPooling",
            "mul": "MulPooling",
            "var": "VarPooling",
            "std": "StdPooling",
            "median": "MedianPooling",
            "vpa": "VariancePreservingPooling",
            "quantile": "QuantilePooling",
            "softmax": "SoftmaxPooling",
            "powermean": "PowerMeanPooling",
            "lstm": "LSTMPooling",
            "gru": "GRUPooling",
            "set2set": "Set2SetPooling",
            "sort": "SortPooling",
            "attentive": "AttentivePooling",
            "graphmultiset": "GraphMultisetTransformerPooling",
            "mlp": "MLPPooling",
            "deepsets": "DeepSetsPooling",
            "settransformer": "SetTransformerPooling",
            "lcm": "LCMPooling",
            "multi": "MultiPooling",
            "degreescaler": "DegreeScalerPooling",
            "virtualnode": "VirtualNodePooling",
        }
        for key, class_name in expected.items():
            assert ReadoutRegistry[key] is getattr(readout_module, class_name), (
                f"ReadoutRegistry['{key}'] should resolve to {class_name}"
            )


# ===================================================================
# PyGAggregationWrapper – the custom adapter
# ===================================================================


class TestPyGAggregationWrapperInterface:
    def test_simple_aliases_use_expected_aggregations(self):
        expected = {
            "sum": aggr.SumAggregation,
            "mean": aggr.MeanAggregation,
            "max": aggr.MaxAggregation,
            "min": aggr.MinAggregation,
        }
        for key, aggregation_class in expected.items():
            readout = ReadoutRegistry[key]()
            assert isinstance(readout, PyGAggregationWrapper), (
                f"ReadoutRegistry['{key}'] should be a PyGAggregationWrapper"
            )
            assert isinstance(readout.aggregation, aggregation_class), (
                f"ReadoutRegistry['{key}'] should wrap {aggregation_class.__name__}"
            )

    def test_routes_graph_batch_and_features(self):
        aggregation = MagicMock(return_value=sentinel.output)
        readout = PyGAggregationWrapper(aggregation)
        graph = SimpleNamespace(batch=torch.tensor([0, 0, 1]))
        features = torch.randn(3, 4)

        result = readout(graph, features)

        assert result is sentinel.output
        aggregation.assert_called_once_with(features, graph.batch)


# ===================================================================
# VirtualNodePooling – fully custom implementation
# ===================================================================


class TestVirtualNodePooling:
    def test_extracts_last_node_per_graph(self):
        g1 = Data(x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]))
        g2 = Data(x=torch.tensor([[4.0, 5.0], [6.0, 7.0]]))
        batch = Batch.from_data_list([g1, g2])

        readout = VirtualNodePooling()
        out = readout(batch, batch.x)

        assert out.shape == (2, 2)
        assert torch.allclose(out[0], torch.tensor([2.0, 3.0]))
        assert torch.allclose(out[1], torch.tensor([6.0, 7.0]))
