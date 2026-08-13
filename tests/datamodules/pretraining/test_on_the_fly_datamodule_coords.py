"""Tests for OnTheFlyDataModule coords passthrough + capability probe.

Covers Stage 2 of issue #29 for the multitask wrapper:

- 3D-accepting base + coords → base receives ``coords=`` at collate time
- 2D-only base + coords → single WARNING logged, base does not receive ``coords``
- Any base + no coords → path unchanged, probe never runs
"""

import logging
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import pytest
from torch.utils.data import StackDataset

from matcha.datamodules.pretraining.on_the_fly_datamodule import OnTheFlyDataModule


# ---------------------------------------------------------------------------
# Fake bases — real classes so ``inspect.signature`` sees real parameters
# ---------------------------------------------------------------------------


class _FakeParams(SimpleNamespace):
    """Stand-in for the base ``.params`` object; only ``batch_size`` is used."""


class _FakeBaseBase:
    """Common scaffolding for the two fakes below."""

    def __init__(self, batch_size: int = 4):
        self.params = _FakeParams(batch_size=batch_size)
        self.collate_fn_map: dict = {}
        # Populated on each call to ``generate_features`` so tests can spy.
        self.last_call: dict | None = None

    def _make_features(self, n: int) -> StackDataset:
        # Return a StackDataset with a single dummy tensor per key so the
        # wrapper's ``features.datasets[key][i]`` loop stays happy.
        x = np.arange(n, dtype=np.float32).reshape(n, 1)
        y = np.zeros((n, 1), dtype=np.float32)
        return StackDataset(x=x, y=y)

    def collate_fn(self, batch_dicts: list[dict]) -> dict:
        # Not asserting on this in these tests; return the raw list-of-dicts
        # so callers can still inspect what was produced downstream.
        return {"batch_dicts": batch_dicts}


class FakeBase3D(_FakeBaseBase):
    """Base that declares a ``coords`` kwarg on ``generate_features``."""

    def generate_features(
        self, mol_list, y=None, coords=None, n_jobs=1
    ) -> StackDataset:
        self.last_call = {
            "n_mols": len(mol_list),
            "coords_received": coords is not None,
            "coords": coords,
            "n_jobs": n_jobs,
        }
        return self._make_features(len(mol_list))


class FakeBase2D(_FakeBaseBase):
    """Base without a ``coords`` kwarg (mirrors the classic 2D signature)."""

    def generate_features(self, mol_list, y=None, n_jobs=1) -> StackDataset:
        self.last_call = {
            "n_mols": len(mol_list),
            "coords_received": False,
            "n_jobs": n_jobs,
        }
        return self._make_features(len(mol_list))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def smiles_batch() -> list[str]:
    return ["CCO", "CCN", "CCC"]


@pytest.fixture()
def sparse_y(smiles_batch) -> sp.csr_matrix:
    # 3 mols × 2 targets, mix of 0 / 1 / -1 to exercise the label re-encoding.
    dense = np.array([[1, 0], [-1, 1], [0, -1]], dtype=np.float32)
    return sp.csr_matrix(dense)


@pytest.fixture()
def coords_batch(smiles_batch) -> list[np.ndarray]:
    return [
        np.arange(3, dtype=np.float32).reshape(1, 3) + i
        for i in range(len(smiles_batch))
    ]


def _batch_items(
    smiles: list[str], y_sparse: sp.csr_matrix, coords: list[np.ndarray] | None
) -> list[dict]:
    items = []
    for i, s in enumerate(smiles):
        item = {"smiles": s, "y": y_sparse[i]}
        if coords is not None:
            item["coords"] = coords[i]
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# (a) 3D-accepting base + coords → base receives coords=
# ---------------------------------------------------------------------------


class Test3DBaseWithCoords:
    def test_base_receives_coords(self, smiles_batch, sparse_y, coords_batch):
        base = FakeBase3D()
        dm = OnTheFlyDataModule(base=base)

        batch = _batch_items(smiles_batch, sparse_y, coords_batch)
        dm.collate_fn(batch)

        assert base.last_call is not None
        assert base.last_call["coords_received"] is True
        assert base.last_call["n_mols"] == len(smiles_batch)

        received = base.last_call["coords"]
        assert isinstance(received, list)
        assert len(received) == len(coords_batch)
        for got, want in zip(received, coords_batch):
            assert got.dtype == np.float32
            np.testing.assert_allclose(got, want.astype(np.float32))

        # Probe result cached True; no warning path taken.
        assert dm._base_accepts_coords is True
        assert dm._coords_ignore_warned is False


# ---------------------------------------------------------------------------
# (b) 2D base + coords → single WARNING, base does not receive coords
# ---------------------------------------------------------------------------


class Test2DBaseWithCoords:
    def test_single_warning_and_coords_dropped(
        self, smiles_batch, sparse_y, coords_batch, caplog
    ):
        base = FakeBase2D()
        dm = OnTheFlyDataModule(base=base)

        batch = _batch_items(smiles_batch, sparse_y, coords_batch)

        caplog.set_level(
            logging.WARNING,
            logger="matcha.datamodules.pretraining.on_the_fly_datamodule",
        )
        dm.collate_fn(batch)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "coords" in r.getMessage().lower()
        ]
        assert len(warnings) == 1
        assert "FakeBase2D" in warnings[0].getMessage()

        assert base.last_call is not None
        assert base.last_call["coords_received"] is False

        # Probe cached False; second call must not re-warn.
        assert dm._base_accepts_coords is False
        assert dm._coords_ignore_warned is True

        caplog.clear()
        dm.collate_fn(batch)
        warnings_second = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "coords" in r.getMessage().lower()
        ]
        assert warnings_second == []


# ---------------------------------------------------------------------------
# (c) Any base + no coords → probe never runs, no coords forwarded
# ---------------------------------------------------------------------------


class TestNoCoordsPathUnchanged:
    def test_probe_not_triggered_when_batch_has_no_coords(self, smiles_batch, sparse_y):
        base = FakeBase3D()
        dm = OnTheFlyDataModule(base=base)

        batch = _batch_items(smiles_batch, sparse_y, coords=None)
        dm.collate_fn(batch)

        # Base still called, but without coords.
        assert base.last_call is not None
        assert base.last_call["coords_received"] is False

        # Probe cache untouched — path is byte-identical to today's behaviour.
        assert dm._base_accepts_coords is None
        assert dm._coords_ignore_warned is False

    def test_no_warning_for_2d_base_without_coords(
        self, smiles_batch, sparse_y, caplog
    ):
        base = FakeBase2D()
        dm = OnTheFlyDataModule(base=base)

        batch = _batch_items(smiles_batch, sparse_y, coords=None)
        caplog.set_level(
            logging.WARNING,
            logger="matcha.datamodules.pretraining.on_the_fly_datamodule",
        )
        dm.collate_fn(batch)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "coords" in r.getMessage().lower()
        ]
        assert warnings == []
        assert dm._base_accepts_coords is None
