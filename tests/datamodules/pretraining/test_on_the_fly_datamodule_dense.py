"""Collate parity tests for the sparse vs dense label paths (issue #31 stage 4).

The pretraining prep pipeline now emits either a ``scipy.sparse.csr_matrix``
(sparse mode: ``0`` = missing, ``-1`` = classification negative) or a 2D
``numpy.ndarray`` (dense mode: ``NaN`` = missing, ``{0, 1}`` = classes).
``OnTheFlyDataModule.collate_fn`` branches on ``scipy.sparse.issparse`` and
both paths must surface an identical ``(B, T)`` tensor with the same NaN
mask and the same non-NaN values so downstream loss / heads are agnostic.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from torch.utils.data import StackDataset

from matcha.datamodules.pretraining.on_the_fly_datamodule import OnTheFlyDataModule


# ---------------------------------------------------------------------------
# Fake base — same collate contract as the coords test's ``FakeBase2D`` but
# it captures the dense ``y`` block that reaches ``generate_features`` so we
# can compare it across the two label layouts.
# ---------------------------------------------------------------------------


class _FakeParams(SimpleNamespace):
    """Stand-in for the base ``.params`` object; only ``batch_size`` is used."""


class CapturingBase:
    """Base that records the ``y`` block passed to ``generate_features``."""

    def __init__(self, batch_size: int = 4):
        self.params = _FakeParams(batch_size=batch_size)
        self.collate_fn_map: dict = {}
        self.last_y: np.ndarray | None = None

    def generate_features(self, mol_list, y=None, n_jobs=1) -> StackDataset:
        self.last_y = np.asarray(y)
        n = len(mol_list)
        x = np.arange(n, dtype=np.float32).reshape(n, 1)
        return StackDataset(x=x, y=self.last_y.astype(np.float32, copy=False))

    def collate_fn(self, batch_dicts: list[dict]) -> dict:
        return {"batch_dicts": batch_dicts}


# ---------------------------------------------------------------------------
# Fixtures — same three-molecule batch expressed in both label layouts.
# ---------------------------------------------------------------------------


@pytest.fixture()
def smiles_batch() -> list[str]:
    return ["CCO", "CCN", "CCC"]


@pytest.fixture()
def dense_label_grid() -> np.ndarray:
    """The canonical label grid used by both layouts.

    Rows are molecules, columns are tasks. ``NaN`` marks missing entries;
    non-missing values are classification classes in ``{0, 1}``.
    """
    return np.array(
        [
            [1.0, np.nan],
            [np.nan, 1.0],
            [0.0, np.nan],
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def sparse_label_rows(dense_label_grid: np.ndarray) -> list[sp.csr_matrix]:
    """Sparse-mode row-encoded equivalent of ``dense_label_grid``.

    Sparse encoding: ``NaN → 0`` (missing), ``0 → -1`` (classification
    negative), ``1 → 1`` (classification positive). This is exactly what
    ``prepare_dataset.py`` produces in sparse mode.
    """
    encoded = np.where(np.isnan(dense_label_grid), 0.0, dense_label_grid)
    encoded = np.where(encoded == 0.0, -1.0, encoded)
    # But keep original NaN positions as sparse zeros (unchanged from the
    # first ``where``). Cast per-row into a csr_matrix so item["y"] mirrors
    # what ``OnTheFlyDataset`` yields at the wrapper's collate boundary.
    encoded = np.where(np.isnan(dense_label_grid), 0.0, encoded).astype(np.float32)
    return [sp.csr_matrix(encoded[i : i + 1]) for i in range(encoded.shape[0])]


@pytest.fixture()
def dense_label_rows(dense_label_grid: np.ndarray) -> list[np.ndarray]:
    """Dense-mode per-row equivalent of ``dense_label_grid``.

    ``OnTheFlyDataset.__getitem__`` yields ``self.y[idx]`` — for a 2D
    ndarray that is a 1D row of shape ``(T,)``.
    """
    return [dense_label_grid[i].copy() for i in range(dense_label_grid.shape[0])]


def _make_batch(smiles: list[str], y_rows: list) -> list[dict]:
    return [{"smiles": s, "y": y_rows[i]} for i, s in enumerate(smiles)]


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------


class TestSparseDenseCollateParity:
    def test_dense_and_sparse_paths_produce_identical_y_block(
        self, smiles_batch, dense_label_grid, dense_label_rows, sparse_label_rows
    ):
        # Sparse path
        sparse_base = CapturingBase()
        sparse_dm = OnTheFlyDataModule(base=sparse_base)
        sparse_dm.collate_fn(_make_batch(smiles_batch, sparse_label_rows))
        y_sparse = sparse_base.last_y

        # Dense path
        dense_base = CapturingBase()
        dense_dm = OnTheFlyDataModule(base=dense_base)
        dense_dm.collate_fn(_make_batch(smiles_batch, dense_label_rows))
        y_dense = dense_base.last_y

        # Both paths must reach the base at all.
        assert y_sparse is not None
        assert y_dense is not None

        # Same shape and dtype.
        assert y_sparse.shape == dense_label_grid.shape
        assert y_dense.shape == dense_label_grid.shape
        assert y_dense.dtype == np.float32

        # NaN mask agrees between the two collate paths and matches the
        # canonical mask from the source grid.
        expected_mask = np.isnan(dense_label_grid)
        np.testing.assert_array_equal(np.isnan(y_sparse), expected_mask)
        np.testing.assert_array_equal(np.isnan(y_dense), expected_mask)

        # Non-missing values are numerically equal across the two paths and
        # match the canonical grid on the non-NaN entries.
        non_nan = ~expected_mask
        np.testing.assert_allclose(
            y_dense[non_nan], dense_label_grid[non_nan], rtol=0, atol=0
        )
        np.testing.assert_allclose(y_sparse[non_nan], y_dense[non_nan], rtol=0, atol=0)

    def test_dense_path_bypasses_sparse_remap(self, smiles_batch, dense_label_rows):
        """Dense rows containing legit ``0.0`` survive collate as ``0.0``.

        Regression guard: the sparse-mode ``0 → NaN`` remap must not run on
        dense inputs, otherwise legitimate classification-negative labels
        would be silently rewritten to missing.
        """
        base = CapturingBase()
        dm = OnTheFlyDataModule(base=base)
        dm.collate_fn(_make_batch(smiles_batch, dense_label_rows))

        y = base.last_y
        assert y is not None
        # Row 2 has a real 0.0 in column 0; ensure it stayed 0.0 (not NaN).
        assert y[2, 0] == 0.0
        assert not np.isnan(y[2, 0])
