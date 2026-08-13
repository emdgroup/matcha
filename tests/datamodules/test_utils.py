"""Tests for utils (CombinedStackDataset, collate helpers)."""

import pytest
import torch
from torch import tensor
from torch.utils.data import StackDataset

from matcha.datamodules.utils import (
    CombinedStackDataset,
    _pad_list,
    collate_fns,
)


# ===================================================================
# _pad_list
# ===================================================================


class TestPadList:
    def test_uniform_sizes(self):
        tensors = [torch.ones(3, 3), torch.ones(3, 3)]
        result = _pad_list(tensors)
        assert result.shape == (2, 3, 3)
        assert (result == 1).all()

    def test_different_sizes(self):
        tensors = [torch.ones(2, 2), torch.ones(4, 4)]
        result = _pad_list(tensors)
        assert result.shape == (2, 4, 4)
        # First tensor should be padded with -1
        assert result[0, 0, 0] == 1
        assert result[0, 3, 3] == -1

    def test_padding_value_is_negative_one(self):
        tensors = [torch.zeros(1, 1), torch.zeros(3, 3)]
        result = _pad_list(tensors)
        assert result[0, 1, 1] == -1
        assert result[0, 0, 0] == 0


# ===================================================================
# CombinedStackDataset
# ===================================================================


class TestCombinedStackDatasetBasic:
    def test_creation_with_disjoint_keys(self):
        ds1 = StackDataset(
            mol_features=tensor([[1.0, 2.0], [3.0, 4.0]]),
            y=tensor([[0.5], [1.5]]),
        )
        ds2 = StackDataset(
            token_ids=tensor([[10, 20], [30, 40]]),
            y=tensor([[0.5], [1.5]]),
        )
        combined = CombinedStackDataset(
            [ds1, ds2], merge_fn={"y": lambda vals: vals[0]}
        )
        assert len(combined) == 2

    def test_getitem(self):
        ds1 = StackDataset(
            mol_features=tensor([[1.0, 2.0], [3.0, 4.0]]),
            y=tensor([[0.5], [1.5]]),
        )
        ds2 = StackDataset(
            token_ids=tensor([[10, 20], [30, 40]]),
            y=tensor([[0.5], [1.5]]),
        )
        combined = CombinedStackDataset(
            [ds1, ds2], merge_fn={"y": lambda vals: vals[0]}
        )
        item = combined[0]
        assert "mol_features" in item
        assert "token_ids" in item
        assert "y" in item

    def test_len(self):
        ds1 = StackDataset(
            mol_features=tensor([[1.0], [2.0], [3.0]]),
            y=tensor([[0.1], [0.2], [0.3]]),
        )
        combined = CombinedStackDataset([ds1])
        assert len(combined) == 3


class TestCombinedStackDatasetMerge:
    def test_duplicate_key_requires_merge_fn(self):
        ds1 = StackDataset(y=tensor([[1.0]]))
        ds2 = StackDataset(y=tensor([[2.0]]))
        with pytest.raises(ValueError, match="merge function"):
            CombinedStackDataset([ds1, ds2])

    def test_merge_fn_applied(self):
        ds1 = StackDataset(y=tensor([[1.0]]))
        ds2 = StackDataset(y=tensor([[2.0]]))
        combined = CombinedStackDataset(
            [ds1, ds2], merge_fn={"y": lambda vals: vals[0]}
        )
        item = combined[0]
        assert item["y"].item() == 1.0


class TestCombinedStackDatasetGetitems:
    def test_getitems_returns_list(self):
        ds1 = StackDataset(
            mol_features=tensor([[1.0], [2.0], [3.0]]),
            y=tensor([[0.1], [0.2], [0.3]]),
        )
        combined = CombinedStackDataset([ds1])
        items = combined.__getitems__([0, 2])
        assert len(items) == 2
        assert items[0]["mol_features"].item() == 1.0
        assert items[1]["mol_features"].item() == 3.0


# ===================================================================
# collate_fns registry
# ===================================================================


class TestCollateFnsRegistry:
    def test_expected_keys_present(self):
        for key in ["mol_features", "graph", "y", "token_ids"]:
            assert key in collate_fns
