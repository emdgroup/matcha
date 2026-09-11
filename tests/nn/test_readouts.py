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
    @pytest.mark.parametrize(
        "key,aggregation_class",
        [
            ("sum", aggr.SumAggregation),
            ("mean", aggr.MeanAggregation),
            ("max", aggr.MaxAggregation),
            ("min", aggr.MinAggregation),
        ],
    )
    def test_simple_alias_uses_expected_aggregation(self, key, aggregation_class):
        readout = ReadoutRegistry[key]()
        assert isinstance(readout, PyGAggregationWrapper)
        assert isinstance(readout.aggregation, aggregation_class)

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
    def test_output_shape(self):
        g1 = Data(x=torch.randn(3, 16))
        g2 = Data(x=torch.randn(2, 16))
        batch = Batch.from_data_list([g1, g2])

        readout = VirtualNodePooling()
        out = readout(batch, batch.x)
        assert out.shape == (2, 16)

    def test_extracts_last_node(self):
        """Should extract the feature of the last node in each graph."""
        g1 = Data(x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]))
        g2 = Data(x=torch.tensor([[4.0, 5.0], [6.0, 7.0]]))
        batch = Batch.from_data_list([g1, g2])

        readout = VirtualNodePooling()
        out = readout(batch, batch.x)
        assert torch.allclose(out[0], torch.tensor([2.0, 3.0]))
        assert torch.allclose(out[1], torch.tensor([6.0, 7.0]))
