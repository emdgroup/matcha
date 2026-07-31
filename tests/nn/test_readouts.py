"""Tests for matcha.nn.readouts – ReadoutRegistry and PyGAggregationWrapper."""

import pytest
import torch


# ===================================================================
# Imports (skip entire module if torch_geometric unavailable)
# ===================================================================

pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data  # noqa: E402

from matcha.nn.readouts import (  # noqa: E402
    ReadoutRegistry,
    PyGAggregationWrapper,
    VirtualNodePooling,
)


# ===================================================================
# Helper to create a small PyG batch
# ===================================================================


@pytest.fixture()
def pyg_batch():
    """Create a small batched PyG graph with 2 graphs (3 + 2 = 5 nodes)."""
    g1 = Data(x=torch.randn(3, 16))
    g2 = Data(x=torch.randn(2, 16))
    batch = Batch.from_data_list([g1, g2])
    return batch


# ===================================================================
# Registry completeness
# ===================================================================


class TestReadoutRegistryKeys:
    EXPECTED_KEYS = [
        "sum",
        "mean",
        "max",
        "min",
        "mul",
        "var",
        "std",
        "median",
        "vpa",
        "softmax",
        "powermean",
        "virtualnode",
    ]

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_key_registered(self, key):
        assert key in ReadoutRegistry, f"'{key}' not found in ReadoutRegistry"


# ===================================================================
# PyGAggregationWrapper – the custom adapter
# ===================================================================


class TestPyGAggregationWrapperInterface:
    """All wrapper-based readouts should accept (graph, x) and return
    (num_graphs, feat_dim), verifying the custom adapter works."""

    SIMPLE_KEYS = ["sum", "mean", "max", "min"]

    @pytest.mark.parametrize("key", SIMPLE_KEYS)
    def test_wrapper_output_shape(self, key, pyg_batch):
        readout = ReadoutRegistry[key]()
        assert isinstance(readout, PyGAggregationWrapper)
        out = readout(pyg_batch, pyg_batch.x)
        assert out.shape == (2, 16)

    @pytest.mark.parametrize("key", SIMPLE_KEYS)
    def test_wrapper_output_is_finite(self, key, pyg_batch):
        readout = ReadoutRegistry[key]()
        out = readout(pyg_batch, pyg_batch.x)
        assert torch.isfinite(out).all()


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
