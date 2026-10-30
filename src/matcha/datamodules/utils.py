"""Utility classes and collation functions for datamodules."""

from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor, cat, stack
from torch.utils.data import Dataset, StackDataset
from torch_geometric.data import Data, Batch


def collate_fn_pyg_graph(graphs: list[Data], collate_fn_map=None) -> Data:
    """Collate function for PyTorch Geometric graphs.

    Batches a list of PyG Data objects into a single batched Data object.
    Handles shortest path distances (spd) and virtual node initialization.

    :param graphs: list of PyG Data objects
    :param collate_fn_map: unused, for compatibility
    :return: batched PyG Data object
    """
    # Collect attributes that need special handling and should be excluded from auto-batching
    exclude_keys = ["spd", "vnode_init"]

    # Use PyG's Batch.from_data_list for batching
    batched_graph = Batch.from_data_list(graphs, exclude_keys=exclude_keys)

    # Handle shortest path distances if present
    if hasattr(graphs[0], "spd") and graphs[0].spd is not None:
        spd_list = [g.spd for g in graphs]
        batched_graph.spd = _pad_list(spd_list)
    else:
        batched_graph.spd = None

    # Handle virtual node initialization if present
    if hasattr(graphs[0], "vnode_init") and graphs[0].vnode_init is not None:
        vnode_init = [g.vnode_init for g in graphs]
        batched_graph.vnode_init = torch.stack(vnode_init)
    else:
        batched_graph.vnode_init = None

    # Apply random sign flipping to Laplacian PE (for sign ambiguity invariance)
    if hasattr(batched_graph, "laplacian_k") and batched_graph.laplacian_k is not None:
        random_signs = (
            torch.randint(
                0,
                2,
                batched_graph.laplacian_k.shape,
                device=batched_graph.laplacian_k.device,
            ).float()
            * 2
            - 1
        )
        batched_graph.laplacian_k = batched_graph.laplacian_k * random_signs

    return batched_graph


def _pad_list(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """Pad each tensor in a list to the maximum size using -1 as padding value.

    :param list[torch.Tensor] tensor_list: list of tensors to pad
    :returns: padded tensors stacked into a single tensor of shape
        (batch_size, max_nodes, max_nodes)
    :rtype: torch.Tensor
    """
    max_nodes = max(t.size(0) for t in tensor_list)
    padded_tensor_list = []
    for t in tensor_list:
        N = t.size(0)
        pad = (0, max_nodes - N, 0, max_nodes - N)
        padded_tensor = torch.nn.functional.pad(t, pad, mode="constant", value=-1)
        padded_tensor_list.append(padded_tensor)
    return torch.stack(padded_tensor_list)


collate_fns: dict[str, Callable] = {
    "mol_features": stack,
    "graph": collate_fn_pyg_graph,
    "y": stack,
    "token_ids": stack,
}


class HandleMissing(Enum):
    """Enum class to handle missing values in the data."""

    RAISE = "raise"
    FILL = "fill"


def concat_tensors_merge_fn(values: list[Tensor], dim=0) -> Tensor:
    """Merge a list of tensors by concatenating them.

    :param values: a list of tensors to concatenate
    :param dim: the dimension along which to concatenate the tensors

    :return: the concatenated tensor
    """
    return cat(values, dim=dim)


class CombinedStackDataset(Dataset):
    """A Dataset that can combine multiple `StackDataset`s.

    This class is useful when you have multiple datasets that you want to combine into a single
    dataset.

    If a key appears in multiple datasets, the values get merged according to merge_fn.

    :param datasets: a list of datasets to combine
    """

    def __init__(
        self, datasets: list[StackDataset], merge_fn: dict[str, Callable] = None
    ):
        """Initialize a combined dataset from multiple StackDatasets.

        :param list[StackDataset] datasets: list of StackDatasets to combine
        :param dict[str, Callable] | None merge_fn: mapping from key name to
            a callable that merges values when a key appears in multiple datasets
        """
        super().__init__()
        self.datasets = datasets
        self.merge_fn = merge_fn or {}
        self._validate_keys()
        self._handle_augmentation()
        self._keys_to_dataset_idx = self._get_keys_to_dataset_idx()

        # (DB) this is not elegant, but I need to know which dataset is which
        # when dealing with this class in an ensemble. the logic is not
        # robust and assumes that only one datamodule of a specific type
        # was used, but for now it will do
        # TODO: find a better way to do this
        self._mapping = {}
        for i, dataset in enumerate(datasets):
            if "mol_features" in dataset[0]:
                self._mapping["tabular"] = i
            if "token_ids" in dataset[0]:
                self._mapping["clm"] = i
            if "graph" in dataset[0]:
                self._mapping["graph"] = i
            if "conformer" in dataset[0]:
                self._mapping["graph3d"] = i

    def _validate_keys(self):
        """Check that all keys that are present in multiple datasets have a merge function."""
        keys = set()
        for dataset in self.datasets:
            for key in dataset[0].keys():
                if key in keys:
                    if key not in self.merge_fn:
                        raise ValueError(
                            f"Key {key} is present in multiple datasets, but no merge function was provided."
                        )
                else:
                    keys.add(key)

    def _handle_augmentation(self):
        """Check that all datasets have the same length."""
        lengths = list(set([len(dataset) for dataset in self.datasets]))
        if len(lengths) > 1:
            aug_factor = int(max(lengths) / min(lengths))

            if len(lengths) > 2:
                raise ValueError(
                    "Multiple augmentation factors are not supported. Please ensure that all datasets that were augmented, had the same augmentation factor."
                )

            for i in range(len(self.datasets)):
                if len(self.datasets[i]) == min(lengths):
                    keys = list(self.datasets[i].datasets.keys())
                    for key in keys:
                        num_dims = self.datasets[i].datasets[key].dim()
                        repeat_pattern = [aug_factor] + [1] * (num_dims - 1)
                        self.datasets[i].datasets[key] = (
                            self.datasets[i].datasets[key].repeat(*repeat_pattern)
                        )

    def _get_keys_to_dataset_idx(self) -> dict[str, list[int]]:
        """Get a mapping from key to dataset index."""
        keys_to_dataset_idx = defaultdict(list)
        for i, dataset in enumerate(self.datasets):
            for key in dataset[0].keys():
                keys_to_dataset_idx[key].append(i)
        return keys_to_dataset_idx

    def __getitem__(self, idx) -> dict[str, Any]:
        """Get the item at the given index.

        :param idx: the index of the item to get

        :return: the item at the given index
        """
        item = {}
        for key, dataset_idxs in self._keys_to_dataset_idx.items():
            values = []
            for dataset_idx in dataset_idxs:
                values.append(self.datasets[dataset_idx][idx][key])
            if len(values) == 1:
                item[key] = values[0]
            else:
                merge_fn = self.merge_fn[key]
                item[key] = merge_fn(values)
        return item

    def __getitems__(self, idx_iterable: list | np.ndarray) -> list[dict[str, Any]]:
        """Return all items in the loader as a list

        :return list[dict[str, Any]]: list with all featurized samples
        """

        return [self.__getitem__(x) for x in idx_iterable]

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        """
        return len(self.datasets[0])
